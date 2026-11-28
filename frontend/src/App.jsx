
import { useState, useEffect, useRef } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

function App() {
  const [chatOpen, setChatOpen] = useState(false);

  const [chatMessage, setChatMessage] = useState('');

  const [chatHistory, setChatHistory] = useState([
    {
      role: 'assistant',
      text: 'Hi! I am CellScan Assistant. Ask me anything about malaria detection, ResNet18, or Grad-CAM.'
    }
  ]);

  const [chatLoading, setChatLoading] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth'
    });
  }, [chatHistory, chatLoading, chatOpen]);

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // --------------------------------------------------
  // IMAGE UPLOAD
  // --------------------------------------------------

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError('');
  };

  // --------------------------------------------------
  // IMAGE PREDICTION
  // --------------------------------------------------

  const handlePrediction = async () => {
    if (!selectedFile) {
      setError('Please upload a blood cell image first.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(
        `${API_BASE_URL}/api/predict`,
        {
          method: 'POST',
          body: formData
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.');
      }

      setResult(data);

    } catch (err) {
      setError(err.message || 'Unable to connect to the backend.');

    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------
  // CELL SCAN AI CHATBOT
  // --------------------------------------------------

  const handleChatSubmit = async (event) => {
    event.preventDefault();

    const message = chatMessage.trim();

    if (!message || chatLoading) return;

    // Convert previous messages into backend format
    const history = chatHistory
      .slice(1)
      .map(({ role, text }) => ({
        role,
        content: text
      }));

    setChatHistory(prev => [
      ...prev,
      { role: 'user', text: message }
    ]);

    setChatMessage('');
    setChatLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message,
          history
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to get a response.');
      }

      setChatHistory(prev => [
        ...prev,
        { role: 'assistant', text: data.reply }
      ]);

    } catch (error) {
      console.error('Chat Error:', error);

      setChatHistory(prev => [
        ...prev,
        {
          role: 'assistant',
          text: 'Sorry, I could not connect to the server. Please try again.'
        }
      ]);

    } finally {
      setChatLoading(false);
    }
  };

  // --------------------------------------------------
  // MAIN UI
  // --------------------------------------------------

  return (
    <div className="app">

      {/* NAVIGATION */}

      <nav className="navbar">
        <a href="#home" className="brand">
          Cell<span>Scan</span> AI
        </a>

        <div className="nav-links">
          <a href="#demo">Live Demo</a>
          <a href="#model">Model</a>

          <button
            className="nav-ai-link"
            onClick={() => setChatOpen(true)}
          >
            Ask AI
          </button>

          <a
            href="https://github.com/HarshitPeriwal24/Cell-Scan-AI"
            target="_blank"
            rel="noreferrer"
          >
            GitHub ↗
          </a>
        </div>
      </nav>

      {/* HERO SECTION */}

      <section className="hero-section" id="home">
        <div className="eyebrow">
          <span></span>
          CNN · RESNET18 · PYTORCH
        </div>

        <h1>
          Detect
          <br />
          <span>malaria</span>
          <br />
          in blood cells
          <br />
          instantly.
        </h1>

        <p className="hero-description">
          Upload a single blood cell or a full thin-smear microscope
          image. Our ResNet18 model finds every red blood cell,
          classifies each as parasitized or uninfected, and explains
          its decision with Grad-CAM.
        </p>

        <div className="hero-buttons">
          <a href="#demo" className="primary-btn">
            Try the detector
          </a>

          <a href="#model" className="secondary-btn">
            How it works
          </a>
        </div>
      </section>

      {/* LIVE DETECTOR */}

      <section className="demo-section" id="demo">
        <div className="section-label">LIVE DEMO</div>

        <h2>Upload a cell image</h2>

        <p>
          Upload a single cropped cell or a whole Giemsa-stained thin
          blood smear field. Full slides are split into individual cells
          and each one is classified.
        </p>

        <div className="detector-placeholder">

          {/* LEFT: IMAGE UPLOAD */}

          <div className="upload-panel">

            <div className="upload-icon">↑</div>

            <h3>Cell image detector</h3>

            <p>
              Select a blood cell microscopy image to analyse it
              using our ResNet18 model.
            </p>

            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />

            {preview && (
              <div className="image-preview">
                <h3>Selected Image</h3>

                <img
                  src={preview}
                  alt="Selected blood cell"
                />
              </div>
            )}

            <button
              className="primary-btn"
              onClick={handlePrediction}
              disabled={!selectedFile || loading}
            >
              {loading ? 'Analysing...' : 'Detect Malaria'}
            </button>

            {error && (
              <p className="error-message">{error}</p>
            )}

          </div>

          {/* RIGHT: PREDICTION RESULT */}

          <div className="result-panel">

            {result ? (
              <div className="prediction-result">

                <h3>Prediction Result</h3>

                <div
                  className={`prediction-status ${
                    result.prediction === 'Parasitized'
                      ? 'status-parasitized'
                      : 'status-uninfected'
                  }`}
                >
                  <span className="status-dot"></span>
                  {result.prediction}
                </div>

                <p className="confidence-text">
                  Confidence: {Number(result.confidence).toFixed(2)}%
                </p>

                {result.mode === 'slide' && (
                  <div className="slide-summary">
                    <p>
                      <strong>{result.cells_parasitized}</strong> of{' '}
                      <strong>{result.cells_analyzed}</strong> detected cells
                      look parasitized (parasitemia{' '}
                      {Number(result.parasitemia).toFixed(1)}%).
                    </p>

                    {result.annotated_url && (
                      <img
                        src={`${API_BASE_URL}${result.annotated_url}`}
                        alt="Detected cells: red = parasitized, green = uninfected"
                        className="gradcam-image"
                      />
                    )}

                    <p className="slide-legend">
                      Red boxes = parasitized, green = uninfected.
                      Probabilities below are for the most suspicious cell.
                    </p>
                  </div>
                )}

                <div className="probability-section">
                  <h4>Class Probabilities</h4>

                  <div className="probability-item">
                    <div className="probability-label">
                      <span>Parasitized</span>
                      <span>
                        {Number(
                          result.probabilities?.Parasitized ?? 0
                        ).toFixed(2)}%
                      </span>
                    </div>

                    <div className="probability-track">
                      <div
                        className="probability-fill parasitized-fill"
                        style={{
                          width: `${Number(
                            result.probabilities?.Parasitized ?? 0
                          )}%`
                        }}
                      />
                    </div>
                  </div>

                  <div className="probability-item">
                    <div className="probability-label">
                      <span>Uninfected</span>
                      <span>
                        {Number(
                          result.probabilities?.Uninfected ?? 0
                        ).toFixed(2)}%
                      </span>
                    </div>

                    <div className="probability-track">
                      <div
                        className="probability-fill uninfected-fill"
                        style={{
                          width: `${Number(
                            result.probabilities?.Uninfected ?? 0
                          )}%`
                        }}
                      />
                    </div>
                  </div>
                </div>

                {result.gradcam_url && (
                  <div className="gradcam-section">
                    <h3>Grad-CAM Visualization</h3>

                    <img
                      src={`${API_BASE_URL}${result.gradcam_url}`}
                      alt="Grad-CAM visualization"
                      className="gradcam-image"
                    />
                  </div>
                )}

              </div>
            ) : (
              <div className="empty-result">
                <div className="upload-icon">✦</div>

                <h3>Your analysis will appear here</h3>

                <p>
                  Upload a blood cell image and click Detect Malaria
                  to view the prediction and Grad-CAM visualization.
                </p>
              </div>
            )}

          </div>
        </div>
      </section>

      {/* MODEL SECTION */}

      <section className="model-section" id="model">

        <h2>
          ResNet18, fine-tuned
          <br />
          for binary classification
        </h2>

        <div className="model-content">

          <div className="model-description">

            <p>
              CellScan AI uses a pretrained <strong>ResNet18</strong>
              architecture with its final fully connected layer
              replaced by a two-class classification layer.
            </p>

            <p>
              Input images are resized to <strong>224 × 224</strong>
              pixels and normalized using ImageNet statistics.
              Training uses data augmentation to improve generalization.
            </p>

            <p>
              The model uses PyTorch, the Adam optimizer and
              cross-entropy loss. The best checkpoint is selected
              using validation performance.
            </p>

            <div className="dataset-card">
              <span>DATASET</span>

              <p>
                NIH Malaria Cell Images Dataset — 27,558 images,
                containing 13,779 Parasitized and 13,779 Uninfected
                samples.
              </p>
            </div>

          </div>

          <div className="pipeline">
            {[
              ['01', 'Input Image', 'RGB microscopy image'],
              ['02', 'Resize + Normalize', '224 × 224 pixels'],
              ['03', 'ResNet18 Backbone', 'ImageNet pretrained'],
              ['04', 'Custom FC Layer', '512 → 2 classes'],
              ['05', 'Softmax Output', 'Parasitized / Uninfected']
            ].map((item) => (
              <div className="pipeline-item" key={item[0]}>
                <span className="step-number">{item[0]}</span>
                <strong>{item[1]}</strong>
                <small>{item[2]}</small>
              </div>
            ))}
          </div>

        </div>
      </section>

      {/* FLOATING CELL SCAN AI CHATBOT */}

      {!chatOpen && (
        <button
          className="chat-toggle"
          onClick={() => setChatOpen(true)}
          aria-label="Open CellScan Assistant"
        >
          ✦
        </button>
      )}

      {chatOpen && (
        <div className="chat-widget">

          {/* CHAT HEADER */}

          <div className="chat-header">
            <div className="chat-header-info">
              <span className="online-dot"></span>

              <div>
                <h3>CellScan Assistant</h3>
                <small>Ask me about malaria detection</small>
              </div>
            </div>

            <button
              className="chat-close"
              onClick={() => setChatOpen(false)}
            >
              ×
            </button>
          </div>

          {/* CHAT MESSAGES */}

          <div className="chat-messages">

            {/* SUGGESTED QUESTIONS */}

            {chatHistory.length === 1 && (
              <div className="quick-questions">
                <p>Suggested questions</p>

                <button
                  onClick={() => setChatMessage("What is ResNet18?")}
                >
                  What is ResNet18?
                </button>

                <button
                  onClick={() => setChatMessage("How does Grad-CAM work?")}
                >
                  How does Grad-CAM work?
                </button>

                <button
                  onClick={() => setChatMessage("What dataset was used?")}
                >
                  What dataset was used?
                </button>

                <button
                  onClick={() =>
                    setChatMessage("What are the limitations of this model?")
                  }
                >
                  What are the limitations of this model?
                </button>
              </div>
            )}

            {/* DISPLAY CHAT HISTORY */}

            {chatHistory.map((message, index) => (
              <div
                key={index}
                className={`chat-message ${message.role}`}
              >
                <ReactMarkdown>
                  {message.text}
                </ReactMarkdown>
              </div>
            ))}

            {/* TYPING INDICATOR */}

            {chatLoading && (
              <div className="chat-message assistant">
                CellScan is typing...
              </div>
            )}

            {/* AUTO SCROLL */}

            <div ref={messagesEndRef} />

          </div>

          {/* CHAT INPUT */}

          <form
            className="chat-form"
            onSubmit={handleChatSubmit}
          >
            <input
              type="text"
              placeholder="Ask something..."
              value={chatMessage}
              onChange={(event) => setChatMessage(event.target.value)}
              disabled={chatLoading}
            />

            <button
              type="submit"
              disabled={!chatMessage.trim() || chatLoading}
            >
              ↑
            </button>
          </form>

        </div>
      )}

      {/* FOOTER */}

      <footer className="footer">
        <a href="#home" className="brand">
          Cell<span>Scan</span> AI
        </a>

        <p>Built with PyTorch · ResNet18 · FastAPI · React</p>

        <a href="#home">Back to top ↑</a>
      </footer>

    </div>
  );
}

export default App;
