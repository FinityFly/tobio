# Tobio backend – Docker & Render

## Run Docker locally

From the **repo root**:

```bash
# Build the image (context must be app/backend)
docker build -t tobio-backend -f app/backend/Dockerfile app/backend

# Run the container (port 8000)
docker run -p 8000:8000 -e PORT=8000 tobio-backend
```

Or from **inside `app/backend`**:

```bash
cd app/backend
docker build -t tobio-backend .
docker run -p 8000:8000 -e PORT=8000 tobio-backend
```

Then open `http://localhost:8000`. If you use env vars (e.g. `API_USERNAME`), pass them:

```bash
docker run -p 8000:8000 -e PORT=8000 -e API_USERNAME=myuser -e API_PASSWORD=mypass tobio-backend
```

---

## Render – where to set Docker and paths

Render uses **“Language”** (or **“Runtime”**) for the type of build, not “Environment.” The **Dockerfile path** and **Root Directory** live under **Settings → Build & Deploy**.

### 1. Creating a **new** Web Service

1. Dashboard → **New +** → **Web Service**.
2. Connect your repo and branch.
3. On the create screen, find the **Language** (or **Runtime**) dropdown.  
   **Choose “Docker”** (not Python). That’s what turns on Dockerfile-based builds.
4. Lower on the same screen you should see **Dockerfile Path**.  
   - If the Dockerfile is in the repo root, leave it blank or `Dockerfile`.  
   - For this backend it’s in a subfolder, so **after** setting Root Directory (next step), you’ll use a path relative to that.
5. Under **Build & Deploy** (or in **Advanced**), set **Root Directory** to `app/backend`.  
   - If **Dockerfile Path** is relative to the repo root, use **`app/backend/Dockerfile`**.  
   - If Render says it’s relative to Root Directory, use **`Dockerfile`** (since root is already `app/backend`).
6. **Build Command** / **Start Command** / **Pre-Deploy Command**: leave empty. The Dockerfile defines the build and `CMD` runs the server.
7. Add env vars (e.g. `API_USERNAME`, `API_PASSWORD`) in the **Environment** section, then deploy.

### 2. You **already have** a Web Service (e.g. Python)

1. Open the service in the dashboard.
2. Go to **Settings** (left sidebar or tab).
3. Scroll to **Build & Deploy**.
4. **Root Directory**  
   - Click **Edit** next to **Root Directory**.  
   - Set it to **`app/backend`** and save.  
   - Paths below are relative to this.
5. **Dockerfile Path**  
   - You’ll only see this if the service is set to **Docker**.  
   - If you currently have **Python** (or another runtime), switch the service to Docker:
     - Some accounts have a **“Runtime”** or **“Language”** (or similar) in Build & Deploy or in the main service settings. Change it to **Docker**.  
     - If there’s no way to switch an existing service to Docker, create a **new** Web Service, choose **Docker** when connecting the repo, then in that new service set Root Directory to `app/backend` and Dockerfile Path to `Dockerfile` (or `app/backend/Dockerfile` if the field is relative to repo root).
6. For **Dockerfile Path**:
   - If the build context is “Root Directory” (`app/backend`): use **`Dockerfile`**.
   - If the build context is “Repository root”: use **`app/backend/Dockerfile`**.

### 3. Build / Start / Pre-deploy commands

- **Build Command**: leave blank (Docker build uses the Dockerfile).
- **Start Command**: leave blank (image uses the Dockerfile `CMD`: `uvicorn api:app --host 0.0.0.0 --port $PORT`).
- **Pre-Deploy Command**: leave blank.

If you ever need to override the start command, use:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

### 4. If “Docker” or “Dockerfile Path” never appears

Then your plan may be **“native”** (e.g. Python), not Docker. In that case:

- **Root Directory**: `app/backend`
- **Build Command**: `pip install -r requirements.txt` (or what you use locally).
- **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`

You won’t get Docker layer caching; builds will re-run `pip install` every time. To use Docker + this Dockerfile, the service must be created or switched to **Language / Runtime: Docker** and have a **Dockerfile Path** set under Build & Deploy.
