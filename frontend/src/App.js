import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Form, Alert, Spinner } from 'react-bootstrap';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function App() {
  const [sequence, setSequence] = useState('');
  const [currencyPair, setCurrencyPair] = useState('USD_INR');
  const [horizon, setHorizon] = useState(1);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [historicalData, setHistoricalData] = useState([]);
  const [showSuccessAlert, setShowSuccessAlert] = useState(false);

  // Fetch model information when component mounts
  useEffect(() => {
    fetchModelInfo();
    
    // Adding some sample historical data for demonstration
    const demoData = [72.85, 73.01, 72.95, 73.12, 73.25, 73.18, 73.30, 73.42, 73.36, 73.50];
    setHistoricalData(demoData);
  }, []);

  const fetchModelInfo = async () => {
    try {
      setModelLoading(true);
      const response = await axios.get(`${API_URL}/model/info`);
      setModelInfo(response.data);
      setModelLoading(false);
    } catch (err) {
      console.error('Error fetching model info:', err);
      setError('Failed to fetch model information');
      setModelLoading(false);
    }
  };

  const handlePrediction = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    try {
      // Parse the input sequence
      const sequenceArray = sequence.split(',').map(num => parseFloat(num.trim()));
      
      if (sequenceArray.some(isNaN)) {
        throw new Error('Invalid sequence format. Please enter comma-separated numbers.');
      }
      
      const response = await axios.post(`${API_URL}/predict`, {
        sequence: sequenceArray,
        currency_pair: currencyPair,
        horizon: parseInt(horizon)
      });
      
      setPrediction(response.data);
      setShowSuccessAlert(true);
      
      // Update historical data with the new prediction
      setHistoricalData([...historicalData, response.data.predicted_value]);
      
      // Reset alert after 3 seconds
      setTimeout(() => {
        setShowSuccessAlert(false);
      }, 3000);
      
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };

  // Chart configuration
  const chartData = {
    labels: [...Array(historicalData.length).keys()].map(i => `Day ${i+1}`),
    datasets: [
      {
        label: `${currencyPair} Rate`,
        data: historicalData,
        fill: true,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        tension: 0.4
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `${currencyPair} Exchange Rate Trend`,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      }
    },
    scales: {
      y: {
        beginAtZero: false,
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  return (
    <Container className="py-5">
      <Row className="mb-4">
        <Col>
          <h1 className="text-center">Forex Rate Prediction</h1>
          <p className="text-center text-muted">
            Predict exchange rates using machine learning
          </p>
        </Col>
      </Row>

      {/* Model Info Card */}
      <Row className="mb-4">
        <Col>
          <Card className="shadow-sm">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">Model Information</h5>
            </Card.Header>
            <Card.Body>
              {modelLoading ? (
                <div className="text-center">
                  <Spinner animation="border" role="status" size="sm" />
                  <span className="ms-2">Loading model information...</span>
                </div>
              ) : modelInfo ? (
                <div>
                  <p><strong>Model Version:</strong> {modelInfo.model_version}</p>
                  <p><strong>Created At:</strong> {new Date(modelInfo.created_at).toLocaleString()}</p>
                  <p><strong>Performance Metrics:</strong></p>
                  <ul>
                    {Object.entries(modelInfo.metrics || {}).map(([key, value]) => (
                      <li key={key}><strong>{key}:</strong> {value.toFixed(4)}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p>No model information available</p>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        {/* Input Form */}
        <Col md={5}>
          <Card className="shadow-sm">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">Make a Prediction</h5>
            </Card.Header>
            <Card.Body>
              <Form onSubmit={handlePrediction}>
                <Form.Group className="mb-3">
                  <Form.Label>Currency Pair</Form.Label>
                  <Form.Select 
                    value={currencyPair}
                    onChange={(e) => setCurrencyPair(e.target.value)}
                    required
                  >
                    <option value="USD_INR">USD/INR</option>
                    <option value="EUR_USD">EUR/USD</option>
                    <option value="GBP_USD">GBP/USD</option>
                  </Form.Select>
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Input Sequence (comma-separated values)</Form.Label>
                  <Form.Control
                    as="textarea"
                    rows={3}
                    placeholder="e.g., 73.25, 73.36, 73.48, 73.55, 73.60"
                    value={sequence}
                    onChange={(e) => setSequence(e.target.value)}
                    required
                  />
                  <Form.Text className="text-muted">
                    Enter historical rates as a comma-separated list
                  </Form.Text>
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Prediction Horizon (days)</Form.Label>
                  <Form.Control
                    type="number"
                    min={1}
                    max={7}
                    value={horizon}
                    onChange={(e) => setHorizon(e.target.value)}
                    required
                  />
                </Form.Group>

                <div className="d-grid">
                  <Button 
                    variant="primary" 
                    type="submit"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <Spinner animation="border" size="sm" className="me-2" />
                        Predicting...
                      </>
                    ) : 'Predict Rate'}
                  </Button>
                </div>
              </Form>

              {showSuccessAlert && (
                <Alert variant="success" className="mt-3">
                  Prediction successful!
                </Alert>
              )}

              {error && (
                <Alert variant="danger" className="mt-3">
                  {error}
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Col>

        {/* Results and Chart */}
        <Col md={7}>
          <Card className="shadow-sm mb-4">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">Prediction Results</h5>
            </Card.Header>
            <Card.Body>
              {prediction ? (
                <div>
                  <h2 className="text-center mb-4">{prediction.predicted_value.toFixed(4)}</h2>
                  <p className="text-center text-muted">
                    Predicted {currencyPair} rate for {horizon} day(s) ahead
                  </p>
                  
                  <div className="d-flex justify-content-around mt-4">
                    <div className="text-center">
                      <h5>{prediction.confidence_interval.lower_bound.toFixed(4)}</h5>
                      <p className="text-muted">Lower Bound</p>
                    </div>
                    <div className="text-center">
                      <h5>{prediction.confidence_interval.upper_bound.toFixed(4)}</h5>
                      <p className="text-muted">Upper Bound</p>
                    </div>
                  </div>
                  
                  <p className="text-muted mt-3">
                    <small>Prediction made at: {new Date(prediction.prediction_timestamp).toLocaleString()}</small>
                  </p>
                </div>
              ) : (
                <p className="text-center text-muted">
                  Submit the form to see prediction results
                </p>
              )}
            </Card.Body>
          </Card>

          <Card className="shadow-sm">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">Historical Trend</h5>
            </Card.Header>
            <Card.Body>
              <div style={{ height: '300px' }}>
                <Line data={chartData} options={chartOptions} />
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default App;