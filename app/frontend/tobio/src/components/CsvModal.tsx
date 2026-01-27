import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';

interface CsvModalProps {
  csvData: string;
  onClose: () => void;
}

const CsvModal: React.FC<CsvModalProps> = ({ csvData, onClose }) => {
  const [parsedData, setParsedData] = useState<string[][]>([]);

  useEffect(() => {
    const parseCsv = () => {
      Papa.parse(csvData, {
        complete: (result) => {
          setParsedData(result.data as string[][]);
        },
        header: false,
      });
    };

    if (csvData) {
      parseCsv();
    }
  }, [csvData]);

  if (!csvData) {
    return null;
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          &times;
        </button>
        {parsedData.length > 0 ? (
          <table>
            <thead>
              <tr>
                {parsedData[0].map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parsedData.slice(1).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>Loading or invalid CSV data...</p>
        )}
      </div>
    </div>
  );
};

export default CsvModal;