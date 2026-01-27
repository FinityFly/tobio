import cv2
import numpy as np

class CourtEstimator:
    def __init__(self, court_class_id=1, approx_epsilon_factor=0.02):
        self.court_class_id = court_class_id
        self.epsilon_factor = approx_epsilon_factor

    def predict(self, results):
        if results.masks is None:
            return self._empty_result("not_found")

        raw_contours = []
        confidences = []
        # get court mask
        for i, class_id_tensor in enumerate(results.boxes.cls):
            if int(class_id_tensor.item()) == self.court_class_id:
                mask_points = results.masks.xy[i].astype(np.int32)
                raw_contours.append(mask_points)
                confidences.append(results.boxes.conf[i].item())
        if not raw_contours:
            return self._empty_result("not_found")

        final_confidence = max(confidences)
        all_points = np.vstack(raw_contours)

        # convex hull and approximate polygon
        hull = cv2.convexHull(all_points)
        perimeter = cv2.arcLength(hull, True)
        epsilon = self.epsilon_factor * perimeter
        approx_poly = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)

        final_corners = self._smart_reduce_to_4_points(approx_poly)
        status = "success" if len(final_corners) == 4 else "partial"
        
        return {
            "corners": final_corners.tolist(),
            "confidence": round(final_confidence, 4),
            "status": status
        }

    def _empty_result(self, status):
        return {
            "corners": [],
            "confidence": 0.0,
            "status": status
        }

    def _smart_reduce_to_4_points(self, points):
        poly = [tuple(p) for p in points]
        if len(poly) <= 4:
            return np.array(poly, dtype=np.int32)

        while len(poly) > 4:
            min_dist_sq = float('inf')
            min_index = -1
            # find shortest edge
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                # dx^2 + dy^2
                d_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    min_index = i
            # get neighbours, find shortest edge
            idx = min_index
            p_current = poly[idx]
            p_next = poly[(idx + 1) % n]
            p_prev = poly[(idx - 1 + n) % n]   
            p_future = poly[(idx + 2) % n]     
            intersection = self._get_line_intersection(p_prev, p_current, p_next, p_future)
            if intersection is not None:
                # replace two points with intersection
                if (idx + 1) % n == 0:
                    poly.pop(0)
                    poly.pop(-1)
                    poly.append(intersection)
                else:
                    poly.pop(idx + 1) 
                    poly.pop(idx)     
                    poly.insert(idx, intersection) 
            else:
                poly.pop(idx)
        return np.array(poly, dtype=np.int32)

    def _get_line_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if den == 0:
            return None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return int(x), int(y)