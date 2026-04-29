import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step2Preprocessing.css';

function Step2Preprocessing({
  spectralData,
  setSpectralData,
  preprocessingOption,
  setPreprocessingOption,
  onNext,
  onApply,
  onClear,
  isApplied,
  canProceed,
  isMobile
}) {
  const navigate = useNavigate();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [preprocessedData, setPreprocessedData] = useState({
    baseline: [],
    normalization: [],
    both: []
  });
  const [selectedView, setSelectedView] = useState(preprocessingOption || 'both'); // Default view

  // Auto-calculate all preprocessing options when component mounts or data changes
  useEffect(() => {
    if (canProceed && spectralData.originalIntensities.length > 0) {
      calculateAllPreprocessing();
    }
  }, [canProceed, spectralData.originalIntensities]);

  const calculateAllPreprocessing = async () => {
    setIsProcessing(true);
    setError(null);

    try {
      // Calculate all three preprocessing options
      const options = ['baseline', 'normalization', 'both'];
      const results = {};

      for (const option of options) {
        const formData = new FormData();
        formData.append('intensities', JSON.stringify(spectralData.originalIntensities));
        formData.append('preprocessing_option', option);

        const response = await fetch('http://localhost:8000/api/preprocess', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Failed to preprocess with option: ${option}`);
        }

        const data = await response.json();
        results[option] = data.processedSpectrum;
      }

      setPreprocessedData(results);

      // Set the "both" option as the default processed spectrum for next steps
      setSpectralData(prev => ({
        ...prev,
        preprocessedIntensities: results.both
      }));

      // Mark as applied since we auto-calculated
      onApply();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleViewChange = (view) => {
    setSelectedView(view);
    setPreprocessingOption(view);
  };

  const handleNext = () => {
    if (isApplied && preprocessedData.both.length > 0) {
      // Always send the "both" (Baseline Correction + Normalization) to next step
      setSpectralData(prev => ({
        ...prev,
        preprocessedIntensities: preprocessedData.both
      }));
      onNext();
      navigate('/step3');
    }
  };

  if (!canProceed) {
    return (
      <div className="step-container">
        <div className="step-content single-column">
          <div className="info-message">
            <h2>Please complete Step 1 first</h2>
            <p>You need to upload a spectrum file before preprocessing.</p>
            <button className="next-button" onClick={() => navigate('/step1')}>
              Go to Step 1
            </button>
          </div>
        </div>
      </div>
    );
  }

  const getCurrentViewData = () => {
    return preprocessedData[selectedView] || [];
  };

  return (
    <div className="step-container">
      <div className="step-content">
        {/* Chart Panel */}
        <div className="chart-panel">
          <h2>Preprocessed Spectrum</h2>
          <div className="chart-wrapper">
            {isProcessing ? (
              <div className="chart-placeholder">
                <div className="loading-spinner"></div>
                <p>Calculating preprocessing options...</p>
              </div>
            ) : getCurrentViewData().length > 0 ? (
              <SpectrumChart
                wavenumbers={spectralData.wavenumbers}
                originalIntensities={spectralData.originalIntensities}
                processedIntensities={getCurrentViewData()}
              />
            ) : (
              <div className="chart-placeholder">
                <p>Preprocessing will calculate automatically</p>
              </div>
            )}
          </div>
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          <h2>Preprocessing Options</h2>
          <p className="control-description" style={{ fontSize: "13px", marginBottom: "1.5rem", color: "var(--text-secondary)" }}>
            Baseline Correction and Min-Max Normalization is the input for the Denoising Process, other options are available for viewing only.
          </p>

          <div className="view-buttons-group">
            <h3 style={{ fontSize: "1rem", marginBottom: "0.75rem", color: "var(--text-primary)", fontWeight: 800, }}>
              Select View
            </h3>
            <div className="view-buttons" >
              <button
                className={`view-button ${selectedView === 'baseline' ? 'selected' : ''}`}
                onClick={() => handleViewChange('baseline')}
                disabled={isProcessing || preprocessedData.baseline.length === 0}
              >
                Baseline Correction
              </button>
              <button
                className={`view-button ${selectedView === 'normalization' ? 'selected' : ''}`}
                onClick={() => handleViewChange('normalization')}
                disabled={isProcessing || preprocessedData.normalization.length === 0}
              >
                Min-Max Normalization
              </button>
              <button
                className={`view-button ${selectedView === 'both' ? 'selected' : ''}`}
                onClick={() => handleViewChange('both')}
                disabled={isProcessing || preprocessedData.both.length === 0}
              >
                Baseline Correction and Min-Max Normalization
              </button>
            </div>
          </div>

          {error && <p className="error-text">{error}</p>}

          <button
            className="next-button"
            onClick={handleNext}
            disabled={!isApplied || preprocessedData.both.length === 0}
          >
            NEXT STEP
          </button>
        </div>
      </div>
    </div>
  );
}

// Spectrum Chart Component with Comparison and Axis Scales
function SpectrumChart({ wavenumbers, originalIntensities, processedIntensities }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  const xMin = 650;
  const xMax = 4000;

  const allIntensities = [...originalIntensities, ...processedIntensities];
  const yMin = Math.min(...allIntensities);
  const yMax = Math.max(...allIntensities);

  const xScale = (x) =>
    padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);

  const yScale = (y) =>
    height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const originalPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(originalIntensities[i])}`)
    .join(' ');

  const processedPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(processedIntensities[i])}`)
    .join(' ');

  // Generate X-axis ticks
  const xTicks = [];
  const xTickCount = 5;
  for (let i = 0; i <= xTickCount; i++) {
    const value = xMin + (i / xTickCount) * (xMax - xMin);
    xTicks.push(value);
  }

  // Generate Y-axis ticks
  const yTicks = [];
  const yTickCount = 5;
  for (let i = 0; i <= yTickCount; i++) {
    const value = yMin + (i / yTickCount) * (yMax - yMin);
    yTicks.push(value);
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg">
      <defs>
        <linearGradient id="inputGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#2563eb" />
          <stop offset="100%" stopColor="#60a5fa" />
        </linearGradient>
        <linearGradient id="processedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#7B2CBF" />
          <stop offset="100%" stopColor="#C77DFF" />
        </linearGradient>
      </defs>

      {/* Grid */}
      <g className="grid" opacity="0.1">
        {yTicks.map((yVal, i) => {
          const y = yScale(yVal);
          return (
            <line
              key={`h-${i}`}
              x1={padding.left}
              y1={y}
              x2={width - padding.right}
              y2={y}
              stroke="#666"
              strokeWidth="1"
            />
          );
        })}
        {xTicks.map((xVal, i) => {
          const x = xScale(xVal);
          return (
            <line
              key={`v-${i}`}
              x1={x}
              y1={padding.top}
              x2={x}
              y2={height - padding.bottom}
              stroke="#666"
              strokeWidth="1"
            />
          );
        })}
      </g>

      {/* Original Spectrum (faded) */}
      <path
        d={originalPath}
        fill="none"
        stroke="url(#inputGradient)"
        strokeWidth="2"
        opacity="1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Processed Spectrum */}
      <path
        d={processedPath}
        fill="none"
        stroke="url(#processedGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Axes */}
      <line
        x1={padding.left}
        y1={height - padding.bottom}
        x2={width - padding.right}
        y2={height - padding.bottom}
        stroke="#666"
        strokeWidth="2"
      />
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={height - padding.bottom}
        stroke="#666"
        strokeWidth="2"
      />

      {/* X-axis Ticks and Labels */}
      {xTicks.map((value, i) => {
        const x = xScale(value);
        return (
          <g key={`x-tick-${i}`}>
            <line
              x1={x}
              y1={height - padding.bottom}
              x2={x}
              y2={height - padding.bottom + 6}
              stroke="#666"
              strokeWidth="2"
            />
            <text
              x={x}
              y={height - padding.bottom + 20}
              textAnchor="middle"
              fill="#999"
              fontSize="11"
            >
              {value.toFixed(0)}
            </text>
          </g>
        );
      })}

      {/* Y-axis Ticks and Labels */}
      {yTicks.map((value, i) => {
        const y = yScale(value);
        return (
          <g key={`y-tick-${i}`}>
            <line
              x1={padding.left - 6}
              y1={y}
              x2={padding.left}
              y2={y}
              stroke="#666"
              strokeWidth="2"
            />
            <text
              x={padding.left - 10}
              y={y + 4}
              textAnchor="end"
              fill="#999"
              fontSize="11"
            >
              {value.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      <g transform="translate(460, 30)">
        <line x1="0" y1="0" x2="30" y2="0" stroke="#2563eb" strokeWidth="2"/>
        <text x="35" y="5" fill="#999" fontSize="12">Input</text>

        <line x1="0" y1="15" x2="30" y2="15" stroke="#7B2CBF" strokeWidth="2"/>
        <text x="35" y="20" fill="#999" fontSize="12">Preprocessed</text>
      </g>

      {/* Axis Labels */}
      <text x={width / 2} y={height - 10} textAnchor="middle" fill="#999" fontSize="14">
        Wavenumber (cm⁻¹)
      </text>
      <text x={20} y={height / 2} textAnchor="middle" fill="#999" fontSize="14"
        transform={`rotate(-90, 20, ${height / 2})`}>
        Intensity
      </text>
    </svg>
  );
}

export default Step2Preprocessing;
