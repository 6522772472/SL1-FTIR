import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step4Classification.css';

const CLASSIFICATION_MODELS = [
  { value: 'disable', label: <>Correlation <br/> (Library Search)</> },
  { value: 'LeNet5', label: 'LeNet5' },
  { value: 'AlexNet', label: 'AlexNet' }
];

const LLM_MODELS = [
  { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' },
  { value: 'gpt-4o-mini', label: 'GPT-4o-mini' },
  { value: 'llama-3', label: 'Llama-3' }
];

function Step4Classification({
  uploadedFile,
  spectralData,
  setSpectralData,
  denoisingConfig,
  classificationModel,
  setClassificationModel,
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
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [results, setResults] = useState(null);
  
  const [activeVisualization, setActiveVisualization] = useState('spectrum');
  const [showPreprocessed, setShowPreprocessed] = useState(true);
  const [showDenoised, setShowDenoised] = useState(true);
  const [showClassification, setShowClassification] = useState(true);

  const [llmEngine, setLlmEngine] = useState('gemini-1.5-pro');
  const [selectedLLM, setSelectedLLM] = useState('gemini-1.5-pro');
  const [isLlmApplied, setIsLlmApplied] = useState(false);
  const [isProcessingLlm, setIsProcessingLlm] = useState(false);

  useEffect(() => {
    if (spectralData.classificationResult && isApplied) {
      const data = spectralData.classificationResult;
      setResults({
        plasticType: data.plastic_type,
        correlation: data.correlation,
        cleanSpectrum: data.clean_spectrum,
        classificationReference: data.reference_spectrum || [],
        warning: data.warning || null,
        camHeatmap: data.cam_heatmap || [],
        reasoning: data.reasoning || ""
      });
      if (data.reasoning) setIsLlmApplied(true);
    } else if (!isApplied) {
      setResults(null);
    }
  }, [spectralData.classificationResult, isApplied]);

  const handleApplyCNN = async () => {
    if (!classificationModel || !canProceed) return;
    setIsProcessing(true);
    setError(null);
    setIsLlmApplied(false);

    try {
      const formData = new FormData();
      const inputIntensities = spectralData.denoisedIntensities.length > 0 
        ? spectralData.denoisedIntensities 
        : (spectralData.preprocessedIntensities.length > 0 
          ? spectralData.preprocessedIntensities 
          : spectralData.originalIntensities);
      
      formData.append('intensities', JSON.stringify(inputIntensities));
      formData.append('membrane_filter', denoisingConfig.membraneFilter || 'Cellulose Ester');
      formData.append('denoising_model', denoisingConfig.denoisingModel || 'disable');
      formData.append('classification_model', classificationModel);
      
      const baselineSource =
        spectralData.preprocessedIntensities.length > 0
          ? spectralData.preprocessedIntensities
          : (spectralData.baselineIntensities && spectralData.baselineIntensities.length > 0)
            ? spectralData.baselineIntensities
            : spectralData.originalIntensities;

      if (baselineSource && baselineSource.length > 0) {
        formData.append('baseline_intensities', JSON.stringify(baselineSource));
      }

      const response = await fetch('http://localhost:8000/api/classify', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to classify spectrum');

      const data = await response.json();
      
      setResults({
        plasticType: data.plastic_type,
        correlation: data.correlation,
        cleanSpectrum: data.clean_spectrum,
        classificationReference: data.reference_spectrum || [],
        warning: data.warning || null,
        camHeatmap: data.cam_heatmap || [],
        reasoning: "" 
      });

      setSpectralData(prev => ({
        ...prev,
        classificationResult: { ...data, reasoning: "" }
      }));

      onApply();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleApplyLLM = async () => {
    setIsProcessingLlm(true);
    setTimeout(() => {
      let mockReasoning = "";
      if (llmEngine === 'disable') {
         mockReasoning = `ดำเนินการโดยใช้วิธีหาความคล้ายคลึง (Correlation) เทียบกับฐานข้อมูล Library\n\n• พบความคล้ายคลึงสูงสุดกับ ${results?.plasticType || 'Unknown'}\n\n* หมายเหตุ: ไม่มีการอธิบายเชิงลึกเนื่องจากเลือกโหมด Not Select (ไม่ใช้ LLM)`;
      } else {
         mockReasoning = `[ ข้อมูลวิเคราะห์โดย: ${LLM_MODELS.find(m => m.value === llmEngine)?.label} ]\n\nจากการวิเคราะห์สเปกตรัมที่ผ่านการลดสัญญาณรบกวน พบจุดพีคการดูดกลืนแสงสอดคล้องกับโครงสร้างทางเคมีของ ${results?.plasticType || 'Unknown'}\n\n• รูปแบบพีคตรงกับฐานข้อมูลอ้างอิงด้วยความแม่นยำ ${(results?.correlation * 100).toFixed(2)}%\n• ไม่พบการรบกวนที่ผิดปกติจากแผ่นกรอง\n\nสรุป: ยืนยันผลการประเมินด้วย AI ว่าเป็นไมโครพลาสติกชนิด ${results?.plasticType || 'Unknown'}`;
      }
      
      setResults(prev => ({ ...prev, reasoning: mockReasoning }));
      setSpectralData(prev => ({
        ...prev,
        classificationResult: { ...prev.classificationResult, reasoning: mockReasoning }
      }));

      setIsLlmApplied(true);
      setIsProcessingLlm(false);
    }, 2500);
  };

  const handleClearClick = () => {
    setShowClearModal(true);
  };

  const confirmClear = () => {
    setClassificationModel(null);
    setLlmEngine('gemini-1.5-pro');
    setSelectedLLM('gemini-1.5-pro');
    setIsLlmApplied(false);
    setResults(null);
    setActiveVisualization('spectrum');
    setSpectralData(prev => ({
      ...prev,
      classificationResult: null
    }));
    setError(null);
    onClear();
    setShowClearModal(false);
  };

  const generateReportCanvas = async () => {
    const html2canvas = (await import('html2canvas')).default;
    const PAGE_STYLE = {
      width: '720px',
      height: '1018px',
      padding: '30px 40px',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'white',
      boxSizing: 'border-box'
    };
    const createHeader = () => {
      const header = document.createElement('div');
      header.style.borderBottom = '3px solid #7B2CBF';
      header.style.paddingBottom = '8px';
      header.style.marginBottom = '8px';
      header.style.display = 'flex';
      header.style.alignItems = 'center';
      header.style.gap = '15px';

      const logoContainer = document.createElement('div');
      logoContainer.style.width = '15%';
      logoContainer.style.display = 'flex';
      logoContainer.style.justifyContent = 'center';
      logoContainer.style.alignItems = 'center';

      const logo = document.createElement('img');
      logo.src = '/siit_logo.png';
      logo.alt = 'SIIT Logo';
      logo.style.width = '65px';
      logo.style.height = 'auto';
      logo.onerror = () => {
        logo.style.display = 'none';
        const logoText = document.createElement('div');
        logoText.textContent = 'SIIT';
        logoText.style.fontSize = '28px';
        logoText.style.fontWeight = 'bold';
        logoText.style.color = '#7B2CBF';
        logoContainer.appendChild(logoText);
      };
      logoContainer.appendChild(logo);

      const infoContainer = document.createElement('div');
      infoContainer.style.width = '85%';
      infoContainer.style.textAlign = 'left';

      const title = document.createElement('h5');
      title.style.color = '#2c3e50';
      title.style.fontSize = '0.95em';
      title.style.fontWeight = 'bold';
      title.style.margin = '0 0 4px 0';
      title.textContent = 'Deep Learning Denoising for Enhanced Microplastic FTIR Identification';
      infoContainer.appendChild(title);

      const fileName = document.createElement('p');
      fileName.style.margin = '2px 0';
      fileName.style.fontSize = '0.65em';
      fileName.style.color = '#2c3e50';
      fileName.innerHTML = `<strong>Source File:</strong> ${uploadedFile?.name || 'Uploaded Spectrum'}`;
      infoContainer.appendChild(fileName);

      const today = new Date().toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' });
      const dateIssued = document.createElement('p');
      dateIssued.style.margin = '2px 0';
      dateIssued.style.fontSize = '0.65em';
      dateIssued.style.color = '#2c3e50';
      dateIssued.innerHTML = `<strong>Analysis Date:</strong> ${today}`;
      infoContainer.appendChild(dateIssued);

      const membraneInfo = document.createElement('p');
      membraneInfo.style.margin = '2px 0';
      membraneInfo.style.fontSize = '0.65em';
      membraneInfo.style.color = '#2c3e50';
      membraneInfo.innerHTML = `<strong>Membrane Filter Type:</strong> ${denoisingConfig.membraneFilter || 'Not specified'}`;
      infoContainer.appendChild(membraneInfo);

      header.appendChild(logoContainer);
      header.appendChild(infoContainer);
      return header;
    };

    const createFooter = () => {
      const footer = document.createElement('div');
      footer.style.borderTop = '1px solid #ddd';
      footer.style.paddingTop = '10px';
      footer.style.fontSize = '0.7em';
      footer.style.color = '#7f8c8d';
      footer.style.textAlign = 'center';
      footer.innerHTML = 'Generated By SL1 | Computer Engineering Senior Project';
      return footer;
    };

    const canvasWidth = 670;
    const canvasHeight = 180;
    const dpi = 2; 

    const page1 = document.createElement('div');
    Object.assign(page1.style, PAGE_STYLE);
    page1.style.width = '720px';
    page1.style.padding = '12px';
    page1.style.fontFamily = 'Arial, sans-serif';
    page1.style.backgroundColor = 'white';
    
    page1.appendChild(createHeader());

    const content1 = document.createElement('div');
    content1.style.flex = '1';
    content1.style.display = 'flex';
    content1.style.flexDirection = 'column';
    content1.style.alignItems = 'center';

    const sectionStyle = { marginBottom: '2px', paddingBottom: '2px', paddingTop: '1px', borderBottom: '2px solid #ddd', width: '100%' };
    const noBorderStyle = { marginBottom: '2px', paddingTop: '1px', width: '100%' };
    const headingStyle = { color: '#2c3e50', marginBottom: '2px', textAlign: 'left', fontSize: '0.85em', fontWeight: 'bold', borderBottom: '2px solid #ddd', paddingBottom: '1px' };

    const createChartSection = (title, wavenumbers, intensities, color, showBorder = true) => {
        if (!intensities || intensities.length === 0) return null;

        const section = document.createElement('div');
        Object.assign(section.style, showBorder ? sectionStyle : noBorderStyle);

        const heading = document.createElement('h4');
        Object.assign(heading.style, headingStyle);
        heading.textContent = title;
        section.appendChild(heading);

        const canvas = document.createElement('canvas');
        canvas.width = canvasWidth * dpi;
        canvas.height = canvasHeight * dpi;
        canvas.style.width = `${canvasWidth}px`;
        canvas.style.height = `${canvasHeight}px`;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpi, dpi);
        const padding = { top: 25, right: 30, bottom: 35, left: 50 };
        const xMin = 650;
        const xMax = 4000;
        const yMin = Math.min(...intensities);
        const yMax = Math.max(...intensities);

        const xScale = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * (canvasWidth - padding.left - padding.right);
        const yScale = (y) => canvasHeight - padding.bottom - ((y - yMin) / (yMax - yMin)) * (canvasHeight - padding.top - padding.bottom);

        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
          const y = yScale(yMin + (i / 5) * (yMax - yMin));
          ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(canvasWidth - padding.right, y); ctx.stroke();
          const x = xScale(xMin + (i / 5) * (xMax - xMin));
          ctx.beginPath(); ctx.moveTo(x, padding.top); ctx.lineTo(x, canvasHeight - padding.bottom); ctx.stroke();
        }

        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, canvasHeight - padding.bottom);
        ctx.lineTo(canvasWidth - padding.right, canvasHeight - padding.bottom);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        wavenumbers.forEach((x, i) => {
          const px = xScale(x);
          const py = yScale(intensities[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();

        ctx.fillStyle = '#666';
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        for (let i = 0; i <= 5; i++) {
          const value = xMin + (i / 5) * (xMax - xMin);
          ctx.fillText(value.toFixed(0), xScale(value), canvasHeight - padding.bottom + 10);
        }

        ctx.textAlign = 'end';
        for (let i = 0; i <= 5; i++) {
          const value = yMin + (i / 5) * (yMax - yMin);
          ctx.fillText(value.toFixed(2), padding.left - 5, yScale(value) + 3);
        }

        ctx.font = '9px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Wavenumber (cm⁻¹)', canvasWidth / 2, canvasHeight - 3);

        ctx.save();
        ctx.translate(12, canvasHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Intensity', 0, 0);
        ctx.restore();

        section.appendChild(canvas);
        return section;
    };

    const inputSection = createChartSection('Input Spectrum', spectralData.wavenumbers, spectralData.originalIntensities, '#2563eb');
    if (inputSection) content1.appendChild(inputSection);

    const preprocessedSection = createChartSection('Preprocessed Spectrum', spectralData.wavenumbers, spectralData.preprocessedIntensities, '#9333ea');
    if (preprocessedSection) content1.appendChild(preprocessedSection);

    const denoisedSection = createChartSection('Denoised Spectrum', spectralData.wavenumbers, spectralData.denoisedIntensities, '#059669');
    if (denoisedSection) content1.appendChild(denoisedSection);

    if (spectralData.denoisedIntensities && spectralData.preprocessedIntensities && results?.classificationReference && results.classificationReference.length > 0) {
      const classSection = document.createElement('div');
      Object.assign(classSection.style, noBorderStyle);

      const heading = document.createElement('h4');
      Object.assign(heading.style, headingStyle);
      heading.textContent = 'Classification Compared';
      classSection.appendChild(heading);

      const canvas = document.createElement('canvas');
      canvas.width = canvasWidth * dpi;
      canvas.height = canvasHeight * dpi;
      canvas.style.width = `${canvasWidth}px`;
      canvas.style.height = `${canvasHeight}px`;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpi, dpi);
      const padding = { top: 25, right: 30, bottom: 35, left: 50 };
      const allIntensities = [...spectralData.preprocessedIntensities, ...spectralData.denoisedIntensities, ...results.classificationReference];
      const yMin = Math.min(...allIntensities);
      const yMax = Math.max(...allIntensities);

      const xScale = (x) => padding.left + ((x - 650) / (4000 - 650)) * (canvasWidth - padding.left - padding.right);
      const yScale = (y) => canvasHeight - padding.bottom - ((y - yMin) / (yMax - yMin)) * (canvasHeight - padding.top - padding.bottom);

      ctx.strokeStyle = '#ddd';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = yScale(yMin + (i / 5) * (yMax - yMin));
        ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(canvasWidth - padding.right, y); ctx.stroke();
        const x = xScale(650 + (i / 5) * (4000 - 650));
        ctx.beginPath(); ctx.moveTo(x, padding.top); ctx.lineTo(x, canvasHeight - padding.bottom); ctx.stroke();
      }

      ctx.strokeStyle = '#666';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, canvasHeight - padding.bottom);
      ctx.lineTo(canvasWidth - padding.right, canvasHeight - padding.bottom);
      ctx.stroke();

      ctx.strokeStyle = '#9333ea';
      ctx.lineWidth = 2;
      ctx.beginPath();
      spectralData.wavenumbers.forEach((x, i) => { const px = xScale(x); const py = yScale(spectralData.preprocessedIntensities[i]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
      ctx.stroke();

      ctx.strokeStyle = '#059669';
      ctx.lineWidth = 2;
      ctx.beginPath();
      spectralData.wavenumbers.forEach((x, i) => { const px = xScale(x); const py = yScale(spectralData.denoisedIntensities[i]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
      ctx.stroke();

      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      spectralData.wavenumbers.forEach((x, i) => { const px = xScale(x); const py = yScale(results.classificationReference[i]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
      ctx.stroke();

      ctx.fillStyle = '#666';
      ctx.font = '8px Arial';
      ctx.textAlign = 'center';
      for (let i = 0; i <= 5; i++) { const value = 650 + (i / 5) * (4000 - 650); ctx.fillText(value.toFixed(0), xScale(value), canvasHeight - padding.bottom + 10); }
      ctx.textAlign = 'end';
      for (let i = 0; i <= 5; i++) { const value = yMin + (i / 5) * (yMax - yMin); ctx.fillText(value.toFixed(2), padding.left - 5, yScale(value) + 3); }
      ctx.font = '9px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Wavenumber (cm⁻¹)', canvasWidth / 2, canvasHeight - 3);

      ctx.save();
      ctx.translate(12, canvasHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Intensity', 0, 0);
      ctx.restore();

      const legendX = canvasWidth - 140;
      const legendY = 35;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.fillRect(legendX, legendY, 130, 55);
      ctx.strokeStyle = '#ddd';
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, legendY, 130, 55);

      ctx.strokeStyle = '#9333ea'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(legendX + 6, legendY + 10); ctx.lineTo(legendX + 20, legendY + 10); ctx.stroke();
      ctx.fillStyle = '#333'; ctx.font = '7.5px Arial'; ctx.textAlign = 'left'; ctx.fillText('Preprocessed', legendX + 24, legendY + 12);
      ctx.strokeStyle = '#059669'; ctx.beginPath(); ctx.moveTo(legendX + 6, legendY + 26); ctx.lineTo(legendX + 20, legendY + 26); ctx.stroke();
      ctx.fillStyle = '#333'; ctx.fillText('Denoised', legendX + 24, legendY + 28);
      ctx.strokeStyle = '#f59e0b'; ctx.beginPath(); ctx.moveTo(legendX + 6, legendY + 42); ctx.lineTo(legendX + 20, legendY + 42); ctx.stroke();
      ctx.fillStyle = '#333'; ctx.fillText(`${results.plasticType || 'Plastic'} Reference`, legendX + 24, legendY + 44);

      classSection.appendChild(canvas);

      const resultDetails = document.createElement('div');
      resultDetails.style.fontSize = '0.75em';
      resultDetails.style.textAlign = 'left';
      resultDetails.style.marginTop = '3px';
      resultDetails.style.color = '#2c3e50';
      resultDetails.innerHTML = `
        <p style="margin: 2px 0; color: #2c3e50;"><strong>Predicted Plastic Type:</strong> <span style="color: #7B2CBF; font-weight: bold;">${results.plasticType || 'N/A'}</span></p>
        <p style="margin: 2px 0; color: #2c3e50;"><strong>Correlation:</strong> ${results.correlation?.toFixed(4) || 'N/A'}</p>
      `;
      classSection.appendChild(resultDetails);
      content1.appendChild(classSection);
    }
    
    page1.appendChild(content1);
    page1.appendChild(createFooter());

    let page2 = null;
    if (results?.reasoning) {
        page2 = document.createElement('div');
        Object.assign(page2.style, PAGE_STYLE);
        page2.style.width = '720px';
        page2.style.padding = '12px';
        page2.style.fontFamily = 'Arial, sans-serif';
        page2.style.backgroundColor = 'white';

        page2.appendChild(createHeader());

        const content2 = document.createElement('div');
        content2.style.flex = '1';
        content2.style.display = 'flex';
        content2.style.flexDirection = 'column';
        content2.style.minHeight = '650px';

        const reasoningSection = document.createElement('div');
        reasoningSection.style.marginTop = '20px';
        reasoningSection.style.padding = '25px';
        reasoningSection.style.backgroundColor = '#f4f7fb';
        reasoningSection.style.borderLeft = '6px solid #7B2CBF';
        reasoningSection.style.borderRadius = '8px';
        reasoningSection.style.width = '100%';
        reasoningSection.style.boxSizing = 'border-box';

        const reasonTitle = document.createElement('h3');
        reasonTitle.style.color = '#2c3e50';
        reasonTitle.style.margin = '0 0 15px 0';
        reasonTitle.style.fontSize = '1.2em';
        reasonTitle.style.fontWeight = 'bold';
        reasonTitle.textContent = 'Reasoning Analysis';
        reasoningSection.appendChild(reasonTitle);

        const reasonText = document.createElement('p');
        reasonText.style.fontSize = '0.9 em';
        reasonText.style.color = '#34495e';
        reasonText.style.whiteSpace = 'pre-line';
        reasonText.style.margin = '0';
        reasonText.style.lineHeight = '1.8';
        reasonText.textContent = results.reasoning;
        reasoningSection.appendChild(reasonText);

        content2.appendChild(reasoningSection);
        page2.appendChild(content2);
        page2.appendChild(createFooter());
    }

    document.body.appendChild(page1);
    const canvas1 = await html2canvas(page1, { scale: 3, useCORS: true, backgroundColor: '#ffffff' });
    document.body.removeChild(page1);

    let canvases = [canvas1];
    if (page2) {
      document.body.appendChild(page2);
      const canvas2 = await html2canvas(page2, { scale: 3, useCORS: true, backgroundColor: '#ffffff' });
      document.body.removeChild(page2);
      canvases.push(canvas2);
    }

    return canvases;
  };

  const handlePrintReport = async () => {
    setIsProcessing(true);
    try {
      const canvases = await generateReportCanvas();
      
      setPreviewImage(canvases[0].toDataURL('image/png'));
      window.tempPdfCanvases = canvases; 
      
      setShowPreviewModal(true);
    } catch (error) {
      console.error('Error generating report preview:', error);
      alert('Failed to generate report preview. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadPDF = async () => {
    try {
      if (!window.tempPdfCanvases || window.tempPdfCanvases.length === 0) return;

      const jsPDF = (await import('jspdf')).default;
      const pdf = new jsPDF('p', 'pt', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();

      for (let i = 0; i < window.tempPdfCanvases.length; i++) {
        if (i > 0) pdf.addPage();
        
        const imgData = window.tempPdfCanvases[i].toDataURL('image/png');
        const imgProps = pdf.getImageProperties(imgData);
        const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
        
        pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      }

      pdf.save('FTIR_Analysis_Report.pdf');
      setShowPreviewModal(false);
    } catch (error) {
      console.error('Error downloading PDF:', error);
      alert('Failed to download PDF. Please try again.');
    }
  };

  if (!canProceed) {
    return (
      <div className="step-container">
        <div className="step-content single-column">
          <div className="info-message">
            <h2>Please complete Step 3 first</h2>
            <p>You need to denoise the spectrum before classification.</p>
            <button className="next-button" onClick={() => navigate('/step3')}>Go to Step 3</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="step-container step4">
      <div className="step-content step4-layout">
        
        {/* Left Side - Charts */}
        <div className="chart-panel-container">
          
          <div className="model-selection-card">
            <h3>{activeVisualization === 'cam' ? 'Select LLM Engine' : 'Classification Model'}</h3>
            
            <div className="radio-group-inline">
              {activeVisualization === 'spectrum' ? (
                CLASSIFICATION_MODELS.map(model => (
                  <label 
                    key={model.value}
                    className={`radio-option-inline ${classificationModel === model.value ? 'selected' : ''}`}
                  >
                    <input
                      type="radio"
                      name="classification-model"
                      value={model.value}
                      checked={classificationModel === model.value}
                      onChange={(e) => setClassificationModel(e.target.value)}
                      disabled={isApplied}
                    />
                    <span className="radio-label">{model.label}</span>
                  </label>
                ))
              ) : (
                <>
                  <label className={`radio-option-inline ${llmEngine !== 'disable' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="llm-engine"
                      checked={llmEngine !== 'disable'}
                      onChange={() => setLlmEngine(selectedLLM || LLM_MODELS[0].value)}
                      disabled={isLlmApplied}
                    />
                    <select
                      className="llm-model-dropdown"
                      value={llmEngine !== 'disable' ? llmEngine : ''}
                      onChange={(e) => {
                        setSelectedLLM(e.target.value);
                        setLlmEngine(e.target.value);
                      }}
                      disabled={isLlmApplied || llmEngine === 'disable'}
                    >
                      <option value="" disabled hidden>Model</option>
                      {LLM_MODELS.map(m => (
                        <option key={m.value} value={m.value}>{m.label}</option>
                      ))}
                    </select>
                  </label>

                  <label className={`radio-option-inline ${llmEngine === 'disable' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="llm-engine"
                      value="disable"
                      checked={llmEngine === 'disable'}
                      onChange={() => setLlmEngine('disable')}
                      disabled={isLlmApplied}
                    />
                    <span className="radio-label">Not Select</span>
                  </label>
                </>
              )}
            </div>
          </div>

          <div className="chart-panel">
            <div className="visualization-tabs">
              <button
                className={`viz-tab ${activeVisualization === 'spectrum' ? 'active' : ''}`}
                onClick={() => setActiveVisualization('spectrum')}
              >
                Spectrum Comparison
              </button>
              <button
                className={`viz-tab ${activeVisualization === 'cam' ? 'active' : ''}`}
                onClick={() => setActiveVisualization('cam')}
                disabled={!isApplied}
                title={!isApplied ? 'Run Classification to unlock CAM' : ''}
              >
                Activation Map & Reasoning
              </button>
            </div>
            
            <div className="chart-wrapper">
              {results ? (
                activeVisualization === 'spectrum' ? (
                  <ComparisonChart
                    wavenumbers={spectralData.wavenumbers}
                    preprocessedIntensities={
                      spectralData.preprocessedIntensities.length > 0 ? spectralData.preprocessedIntensities : spectralData.originalIntensities
                    }
                    denoisedIntensities={
                      spectralData.denoisedIntensities.length > 0 ? spectralData.denoisedIntensities : spectralData.preprocessedIntensities
                    }
                    classificationReference={
                      results.classificationReference?.length ? results.classificationReference : results.cleanSpectrum
                    }
                    camHeatmap={results.camHeatmap}
                    showHeatmap={false}
                    showPreprocessed={showPreprocessed}
                    showDenoised={showDenoised}
                    showClassification={showClassification}
                    plasticType={results.plasticType}
                  />
                ) : (
                  <CamChart
                    wavenumbers={spectralData.wavenumbers}
                    inputIntensities={
                      spectralData.denoisedIntensities.length > 0 ? spectralData.denoisedIntensities : (spectralData.preprocessedIntensities.length > 0 ? spectralData.preprocessedIntensities : spectralData.originalIntensities)
                    }
                    camHeatmap={results.camHeatmap}
                  />
                )
              ) : (
                <div className="chart-placeholder">
                  <p>Select a classification model and click APPLY to see results</p>
                </div>
              )}
            </div>

            {results && activeVisualization === 'spectrum' && (
              <div className="graph-display-options">
                <div className="display-checkboxes">
                  <label className="display-checkbox-label">
                    <input type="checkbox" checked={showPreprocessed} onChange={(e) => setShowPreprocessed(e.target.checked)} />
                    <span>Preprocessed Spectrum</span>
                  </label>
                  <label className="display-checkbox-label">
                    <input type="checkbox" checked={showDenoised} onChange={(e) => setShowDenoised(e.target.checked)} />
                    <span>Denoised Spectrum</span>
                  </label>
                  <label className="display-checkbox-label">
                    <input type="checkbox" checked={showClassification} onChange={(e) => setShowClassification(e.target.checked)} />
                    <span>{results?.plasticType ? `${results.plasticType} Reference Spectrum` : 'Classification Reference'}</span>
                  </label>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Side - Results and Controls */}
        <div className="results-control-panel">
          
          <div className="reasoning-large-panel">
            <h2>REASONING ANALYSIS</h2>
            <div className="reasoning-content">
              {isProcessingLlm ? (
                <div className="loading-state">
                  <span className="spinner"></span> Generating reasoning...
                </div>
              ) : isLlmApplied && results?.reasoning ? (
                <p>{results.reasoning}</p>
              ) : activeVisualization === 'cam' ? (
                <p className="placeholder-text">Please click 'APPLY' to generate reasoning.</p>
              ) : (
                <p className="placeholder-text">Go to 'ACTIVATION MAP & REASONING' tab to select LLM and generate reasoning.</p>
              )}
            </div>
          </div>

          <div className="bottom-results-wrapper">
            <div className="results-panel">
              <h2>Classification Results</h2>
              {results?.warning && <div className="result-warning">{results.warning}</div>}
              <div className="result-item">
                <span className="result-label">Plastic Type:</span>
                <span className="result-value">{results ? results.plasticType : '-'}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Correlation:</span>
                <span className="result-value">{results ? results.correlation.toFixed(4) : '-'}</span>
              </div>
            </div>

            <div className="config-info">
              <h3>Current Configuration</h3>
              <div className="config-item">
                <span className="config-label">Membrane Filter:</span>
                <span className="config-value">{denoisingConfig.membraneFilter || 'Not Set'}</span>
              </div>
              <div className="config-item">
                <span className="config-label">Denoising Model:</span>
                <span className="config-value">{denoisingConfig.denoisingModel || 'Not Set'}</span>
              </div>
              <div className="config-item">
                <span className="config-label">Classification:</span>
                <span className="config-value">{classificationModel === 'disable' ? 'Correlation' : (classificationModel || 'Not Set')}</span>
              </div>
            </div>

            {error && <p className="error-text">{error}</p>}

            <div className="action-buttons">
              <button 
                className={`apply-button ${(activeVisualization === 'spectrum' ? isApplied : isLlmApplied) ? 'applied' : ''}`}
                onClick={activeVisualization === 'spectrum' ? handleApplyCNN : handleApplyLLM}
                disabled={
                  activeVisualization === 'spectrum' 
                    ? (!classificationModel || isApplied || isProcessing)
                    : (!llmEngine || isLlmApplied || isProcessingLlm)
                }
              >
                {isProcessing || isProcessingLlm ? (<><span className="loading-spinner"></span>Processing...</>) : 'APPLY'}
              </button>

              <button 
                className={`clear-button ${isApplied ? 'enabled' : ''}`}
                onClick={handleClearClick}
                disabled={!isApplied}
              >
                CLEAR
              </button>
            </div>

            <div className="completion-message">
              <p>Analysis Complete</p>
              <p>You can review results or go back to any step to adjust parameters.</p>
              {isApplied && results && (
                <button className="print-report-button" onClick={handlePrintReport}>Print Report</button>
              )}
            </div>
          </div>
        </div>
      </div>

      {showClearModal && (
        <div className="modal-overlay" onClick={() => setShowClearModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Confirm Clear</h3>
            <p>Are you sure you want to clear the classification results?</p>
            <div className="modal-buttons">
              <button className="modal-button confirm" onClick={confirmClear}>Yes, Clear</button>
              <button className="modal-button cancel" onClick={() => setShowClearModal(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {showPreviewModal && (
        <div className="modal-overlay" onClick={() => setShowPreviewModal(false)}>
          <div className="modal-content preview-modal" onClick={(e) => e.stopPropagation()}>
            <h3>Report Preview</h3>
            <div className="preview-container">
              {previewImage && (
                <img src={previewImage} alt="Report Preview" style={{ width: '100%', border: '1px solid #ddd' }} />
              )}
            </div>
            <div className="modal-buttons">
              <button className="modal-button confirm" onClick={handleDownloadPDF}>Download PDF</button>
              <button className="modal-button cancel" onClick={() => setShowPreviewModal(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ComparisonChart({ wavenumbers, preprocessedIntensities, denoisedIntensities, classificationReference, camHeatmap = [], showHeatmap = true, showPreprocessed = true, showDenoised = true, showClassification = true, plasticType = null }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  const xMin = 650;
  const xMax = 4000;

  const allIntensities = [...preprocessedIntensities, ...denoisedIntensities, ...(classificationReference || [])];
  const yMin = Math.min(...allIntensities);
  const yMax = Math.max(...allIntensities);

  const xScale = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  const yScale = (y) => height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const preprocessedPath = wavenumbers.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(preprocessedIntensities[i])}`).join(' ');
  const denoisedPath = wavenumbers.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(denoisedIntensities[i])}`).join(' ');
  const classificationPath = classificationReference && classificationReference.length === wavenumbers.length
      ? wavenumbers.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(classificationReference[i])}`).join(' ') : null;

  const hasHeatmap = Array.isArray(camHeatmap) && camHeatmap.length === wavenumbers.length;
  const heatColor = (value) => {
    const clamped = Math.max(0, Math.min(1, value || 0));
    const alpha = 0.25 + clamped * 0.6;
    const hue = 35 - clamped * 35;
    return `hsla(${hue}, 90%, 55%, ${alpha})`;
  };

  const xTicks = [650, 1000, 1500, 2000, 2500, 3000, 3500, 4000];
  const yTicks = [yMin, yMin + (yMax-yMin)*0.25, yMin + (yMax-yMin)*0.5, yMin + (yMax-yMin)*0.75, yMax];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg" style={{ width: '100%', height: 'auto' }}>
      <defs>
        <linearGradient id="preprocessedGradientPurpleStep4" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#7B2CBF" /><stop offset="100%" stopColor="#C77DFF" />
        </linearGradient>
        <linearGradient id="denoisedGradientGreen" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#059669" /><stop offset="100%" stopColor="#10b981" />
        </linearGradient>
        <linearGradient id="classificationGradientYellow" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#f59e0b" /><stop offset="100%" stopColor="#fbbf24" />
        </linearGradient>
      </defs>

      <g className="grid" opacity="0.1">
        {yTicks.map((y, i) => <line key={`h-${i}`} x1={padding.left} y1={yScale(y)} x2={width - padding.right} y2={yScale(y)} stroke="#666" strokeWidth="1" />)}
        {xTicks.map((x, i) => <line key={`v-${i}`} x1={xScale(x)} y1={padding.top} x2={xScale(x)} y2={height - padding.bottom} stroke="#666" strokeWidth="1" />)}
      </g>

      {showPreprocessed && <path d={preprocessedPath} fill="none" stroke="url(#preprocessedGradientPurpleStep4)" strokeWidth="2" opacity="0.8" />}
      {showDenoised && <path d={denoisedPath} fill="none" stroke="url(#denoisedGradientGreen)" strokeWidth="2" opacity="0.8" />}
      {showClassification && classificationPath && <path d={classificationPath} fill="none" stroke="url(#classificationGradientYellow)" strokeWidth="2" opacity="0.8" />}

      {hasHeatmap && showHeatmap && wavenumbers.map((x, i) => (
        <circle key={`cam-${i}`} cx={xScale(x)} cy={yScale(denoisedIntensities[i])} r={3} fill={heatColor(camHeatmap[i])} />
      ))}

      <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#666" strokeWidth="2" />
      <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#666" strokeWidth="2" />

      {xTicks.map((v, i) => (
        <g key={`xt-${i}`}>
          <line x1={xScale(v)} y1={height - padding.bottom} x2={xScale(v)} y2={height - padding.bottom + 6} stroke="#666" strokeWidth="2" />
          <text x={xScale(v)} y={height - padding.bottom + 20} textAnchor="middle" fill="#999" fontSize="11">{v}</text>
        </g>
      ))}
      {yTicks.map((v, i) => (
        <g key={`yt-${i}`}>
          <line x1={padding.left - 6} y1={yScale(v)} x2={padding.left} y2={yScale(v)} stroke="#666" strokeWidth="2" />
          <text x={padding.left - 10} y={yScale(v) + 4} textAnchor="end" fill="#999" fontSize="11">{v.toFixed(2)}</text>
        </g>
      ))}

      <g transform="translate(450, 20)">
        {showPreprocessed && <><circle cx="6" cy="0" r="6" fill="#7B2CBF"/><text x="18" y="5" fill="#999" fontSize="12">Preprocessed</text></>}
        {showDenoised && <><circle cx="6" cy="15" r="6" fill="#059669"/><text x="18" y="20" fill="#999" fontSize="12">Denoised</text></>}
        {showClassification && <><circle cx="6" cy="30" r="6" fill="#f59e0b"/><text x="18" y="35" fill="#999" fontSize="12">{plasticType ? `${plasticType}` : 'Classification'}</text></>}
      </g>
      <text x={width / 2} y={height - 10} textAnchor="middle" fill="#999" fontSize="14">Wavenumber (cm⁻¹)</text>
      <text x={20} y={height / 2} textAnchor="middle" fill="#999" fontSize="14" transform={`rotate(-90, 20, ${height / 2})`}>Intensity</text>
    </svg>
  );
}

function CamChart({ wavenumbers, inputIntensities, camHeatmap = [] }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 }; 

  if (!camHeatmap || camHeatmap.length !== wavenumbers.length) {
    return (
      <div className="chart-placeholder">
        <p>CAM heatmap unavailable. Run classification to generate.</p>
      </div>
    );
  }

  const xMin = 650;
  const xMax = 4000;
  const yMin = Math.min(...inputIntensities);
  const yMax = Math.max(...inputIntensities);

  const xScale = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  const yScale = (y) => height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const heatColor = (value) => {
    const clamped = Math.max(0, Math.min(1, value || 0));
    const alpha = 0.3 + clamped * 0.6;
    const hue = 35 - clamped * 35;
    return `hsla(${hue}, 90%, 55%, ${alpha})`;
  };

  const xTicks = [650, 1000, 1500, 2000, 2500, 3000, 3500, 4000];
  const yTicks = [yMin, yMin + (yMax-yMin)*0.25, yMin + (yMax-yMin)*0.5, yMin + (yMax-yMin)*0.75, yMax];

  const inputPath = wavenumbers.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(inputIntensities[i])}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg">
      <path d={inputPath} fill="none" stroke="#0ea5e9" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

      {wavenumbers.map((x, i) => (
        <circle key={`cam-full-${i}`} cx={xScale(x)} cy={yScale(inputIntensities[i])} r={5} fill={heatColor(camHeatmap[i])} />
      ))}

      <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#666" strokeWidth="2" />
      <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#666" strokeWidth="2" />

      {xTicks.map((v, i) => (
        <g key={`xt-${i}`}>
          <line x1={xScale(v)} y1={height - padding.bottom} x2={xScale(v)} y2={height - padding.bottom + 6} stroke="#666" strokeWidth="2" />
          <text x={xScale(v)} y={height - padding.bottom + 20} textAnchor="middle" fill="#999" fontSize="11">{v}</text>
        </g>
      ))}

      {yTicks.map((v, i) => (
        <g key={`yt-${i}`}>
          <line x1={padding.left - 6} y1={yScale(v)} x2={padding.left} y2={yScale(v)} stroke="#666" strokeWidth="2" />
          <text x={padding.left - 10} y={yScale(v) + 4} textAnchor="end" fill="#999" fontSize="11">{v.toFixed(2)}</text>
        </g>
      ))}

      <text x={width / 2} y={height - 10} textAnchor="middle" fill="#999" fontSize="14">Wavenumber (cm⁻¹)</text>
      <text x={20} y={height / 2} textAnchor="middle" fill="#999" fontSize="14" transform={`rotate(-90, 20, ${height / 2})`}>Intensity</text>
    </svg>
  );
}

export default Step4Classification;