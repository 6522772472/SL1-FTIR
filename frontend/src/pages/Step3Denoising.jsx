import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step3Denoising.css';

const DENOISING_MODELS = [
  { value: 'disable', label: 'Disable (No Denoising)' },
  { value: 'cae', label: 'CAE (Convolutional Autoencoder)' },
  { value: 'VGG16-Transformer', label: 'VGG16 + Transformer' },
  { value: 'ResNet50-Transformer', label: 'ResNet50 + Transformer' }
];

function Step3Denoising({
  spectralData,
  setSpectralData,
  denoisingConfig,
  setDenoisingConfig,
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
  const [showClearModal, setShowClearModal] = useState(false);

  const handleDenoisingModelChange = (value) => {
    if (!isApplied) {
      setDenoisingConfig(prev => ({ ...prev, denoisingModel: value }));
    }
  };

  const handleApply = async () => {
    // เช็คแค่ว่าเลือก Denoising Model แล้วหรือยัง
    if (!denoisingConfig.denoisingModel || !canProceed) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      const inputIntensities = spectralData.preprocessedIntensities.length > 0 
        ? spectralData.preprocessedIntensities 
        : spectralData.originalIntensities;
      
      formData.append('intensities', JSON.stringify(inputIntensities));
      // ส่งแค่ค่า denoising_model ไปให้ Backend
      formData.append('denoising_model', denoisingConfig.denoisingModel);

      const response = await fetch('http://localhost:8000/api/denoise', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to denoise spectrum');
      }

      const data = await response.json();
      
      setSpectralData(prev => ({
        ...prev,
        denoisedIntensities: data.denoisedSpectrum
      }));

      onApply();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearClick = () => {
    setShowClearModal(true);
  };

  const confirmClear = () => {
    // รีเซ็ตเฉพาะ denoisingModel
    setDenoisingConfig({ denoisingModel: null });
    setSpectralData(prev => ({
      ...prev,
      denoisedIntensities: []
    }));
    setError(null);
    onClear();
    setShowClearModal(false);
  };

  const handleNext = () => {
    if (isApplied) {
      onNext();
      navigate('/step4');
    }
  };

  if (!canProceed) {
    return (
      <div className="step-container">
        <div className="step-content single-column">
          <div className="info-message">
            <h2>Please complete Step 2 first</h2>
            <p>You need to preprocess the spectrum before denoising.</p>
            <button className="next-button" onClick={() => navigate('/step2')}>
              Go to Step 2
            </button>
          </div>
        </div>
      </div>
    );
  }

  // เช็คแค่การเลือก Model อย่างเดียว
  const isSelectionComplete = !!denoisingConfig.denoisingModel;

  return (
    <div className="step-container step3">
      <div className="step-content">
        {/* Chart Panel */}
        <div className="chart-panel">
          <h2>Denoised Spectrum</h2>
          <div className="chart-wrapper">
            {spectralData.denoisedIntensities.length > 0 ? (
              <SpectrumChart 
                wavenumbers={spectralData.wavenumbers}
                preprocessedIntensities={
                  spectralData.preprocessedIntensities.length > 0 
                    ? spectralData.preprocessedIntensities 
                    : spectralData.originalIntensities
                }
                denoisedIntensities={spectralData.denoisedIntensities}
              />
            ) : (
              <div className="chart-placeholder">
                <p>Select denoising model, then click APPLY</p>
              </div>
            )}
          </div>
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          <h2>Denoising Configuration</h2>

          {/* Denoising Model Selection */}
          <div className="radio-group">
            <h3>Denoising Model</h3>
            {DENOISING_MODELS.map(model => (
              <label 
                key={model.value}
                className={`radio-option ${denoisingConfig.denoisingModel === model.value ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="denoising-model"
                  value={model.value}
                  checked={denoisingConfig.denoisingModel === model.value}
                  onChange={(e) => handleDenoisingModelChange(e.target.value)}
                  disabled={isApplied}
                />
                <span className="radio-label">{model.label}</span>
              </label>
            ))}
          </div>

          {error && <p className="error-text">{error}</p>}

          <div className="action-buttons">
            <button 
              className={`apply-button ${isApplied ? 'applied' : ''}`}
              onClick={handleApply}
              disabled={!isSelectionComplete || isApplied || isProcessing}
            >
              {isProcessing ? (
                <>
                  <span className="loading-spinner"></span>
                  Processing...
                </>
              ) : (
                'APPLY'
              )}
            </button>
            <button 
              className={`clear-button ${isApplied ? 'enabled' : ''}`}
              onClick={handleClearClick}
              disabled={!isApplied}
            >
              CLEAR
            </button>
          </div>

          <button 
            className="next-button"
            onClick={handleNext}
            disabled={!isApplied}
          >
            NEXT STEP
          </button>
        </div>
      </div>

      {/* Clear Confirmation Modal */}
      {showClearModal && (
        <div className="modal-overlay" onClick={() => setShowClearModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Confirm Clear</h3>
            <p>Are you sure you want to clear the denoising configuration? This will reset Step 4.</p>
            <div className="modal-buttons">
              <button className="modal-button confirm" onClick={confirmClear}>
                Yes, Clear
              </button>
              <button className="modal-button cancel" onClick={() => setShowClearModal(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Spectrum Chart Component with Comparison
function SpectrumChart({ wavenumbers, preprocessedIntensities, denoisedIntensities }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  const xMin = 650;
  const xMax = 4000;

  const allIntensities = [...preprocessedIntensities, ...denoisedIntensities];
  const yMin = Math.min(...allIntensities);
  const yMax = Math.max(...allIntensities);

  const xScale = (x) =>
    padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);

  const yScale = (y) =>
    height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const preprocessedPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(preprocessedIntensities[i])}`)
    .join(' ');

  const denoisedPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(denoisedIntensities[i])}`)
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
        <linearGradient id="preprocessedGradientPurple" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#7B2CBF" />
          <stop offset="100%" stopColor="#C77DFF" />
        </linearGradient>
        <linearGradient id="denoisedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#059669" />
          <stop offset="100%" stopColor="#10b981" />
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

      {/* Preprocessed Spectrum (purple with opacity 0.7) */}
      <path
        d={preprocessedPath}
        fill="none"
        stroke="url(#preprocessedGradientPurple)"
        strokeWidth="2"
        opacity="0.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Denoised Spectrum (green) */}
      <path
        d={denoisedPath}
        fill="none"
        stroke="url(#denoisedGradient)"
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
      <g transform="translate(440, 30)">
        <circle cx="6" cy="0" r="6" fill="#7B2CBF" opacity="0.7"/>
        <text x="18" y="5" fill="#999" fontSize="12">Preprocessed</text>

        <circle cx="6" cy="15" r="6" fill="#059669"/>
        <text x="18" y="20" fill="#999" fontSize="12">Denoised</text>
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

export default Step3Denoising;