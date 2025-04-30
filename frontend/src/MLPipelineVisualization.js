import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Form, Alert, Spinner, Table, Nav, Tab } from 'react-bootstrap';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

// This URL should match your backend endpoint
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function MLPipelineVisualization() {
  const [pipelineInfo, setPipelineInfo] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState([]);
  const [pipelineMetrics, setPipelineMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchPipelineData();
    // Setting interval to refresh pipeline status every 30 seconds
    const interval = setInterval(() => {
      fetchPipelineStatus();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchPipelineData = async () => {
    setLoading(true);
    try {
      // These endpoints would need to be implemented in your backend
      const infoResponse = await axios.get(`${API_URL}/pipeline/info`);
      const statusResponse = await axios.get(`${API_URL}/pipeline/status`);
      const metricsResponse = await axios.get(`${API_URL}/pipeline/metrics`);
      
      setPipelineInfo(infoResponse.data);
      setPipelineStatus(statusResponse.data.runs || []);
      setPipelineMetrics(metricsResponse.data);
      setError('');
    } catch (err) {
      console.error('Error fetching pipeline data:', err);
      setError('Failed to load pipeline data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const fetchPipelineStatus = async () => {
    try {
      const statusResponse = await axios.get(`${API_URL}/pipeline/status`);
      setPipelineStatus(statusResponse.data.runs || []);
    } catch (err) {
      console.error('Error fetching pipeline status updates:', err);
    }
  };

  const getStatusBadge = (status) => {
    const badgeClass = {
      'running': 'bg-primary',
      'completed': 'bg-success',
      'failed': 'bg-danger',
      'pending': 'bg-warning'
    }[status.toLowerCase()] || 'bg-secondary';
    
    return <span className={`badge ${badgeClass}`}>{status}</span>;
  };

  const renderPipelineStages = () => {
    if (!pipelineInfo || !pipelineInfo.stages) {
      return <Alert variant="info">Pipeline stage information not available</Alert>;
    }

    return (
      <div className="pipeline-visualization mt-3">
        <div className="d-flex justify-content-between align-items-center flex-wrap">
          {pipelineInfo.stages.map((stage, index) => {
            const stageClass = {
              'completed': 'bg-success',
              'running': 'bg-primary',
              'failed': 'bg-danger'
            }[stage.status.toLowerCase()] || 'bg-secondary';
            
            return (
              <div key={stage.name} className="pipeline-stage text-center mb-4">
                <div className={`stage-box p-3 rounded ${stageClass}`}>
                  <h5 className="text-white mb-0">{stage.name}</h5>
                </div>
                <div className="mt-2">
                  {getStatusBadge(stage.status)}
                </div>
                {index < pipelineInfo.stages.length - 1 && (
                  <div className="pipeline-arrow">
                    <span className="arrow-line d-inline-block bg-dark mx-3" style={{ height: '2px', width: '50px' }}></span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderPipelineStatus = () => {
    if (pipelineStatus.length === 0) {
      return <Alert variant="info">No recent pipeline runs available</Alert>;
    }

    return (
      <Table striped bordered hover>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Start Time</th>
            <th>Duration</th>
            <th>Status</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {pipelineStatus.map(run => (
            <tr key={run.run_id}>
              <td>{run.run_id}</td>
              <td>{new Date(run.start_time).toLocaleString()}</td>
              <td>{run.duration}s</td>
              <td>{getStatusBadge(run.status)}</td>
              <td>
                {run.status.toLowerCase() === 'failed' ? (
                  <span className="text-danger">{run.error_message || 'Unknown error'}</span>
                ) : (
                  run.details || 'No details available'
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    );
  };

  const renderMetrics = () => {
    if (!pipelineMetrics) {
      return <Alert variant="info">Pipeline metrics not available</Alert>;
    }

    return (
      <div>
        <Row>
          <Col md={6}>
            <Card className="mb-3">
              <Card.Header>Data Processing Metrics</Card.Header>
              <Card.Body>
                <Table bordered size="sm">
                  <tbody>
                    <tr>
                      <td>Total Records Processed</td>
                      <td>{pipelineMetrics.data_processing.total_records}</td>
                    </tr>
                    <tr>
                      <td>Processing Time</td>
                      <td>{pipelineMetrics.data_processing.processing_time}s</td>
                    </tr>
                    <tr>
                      <td>Records per Second</td>
                      <td>{pipelineMetrics.data_processing.records_per_second}</td>
                    </tr>
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6}>
            <Card className="mb-3">
              <Card.Header>Model Training Metrics</Card.Header>
              <Card.Body>
                <Table bordered size="sm">
                  <tbody>
                    <tr>
                      <td>Training Time</td>
                      <td>{pipelineMetrics.model_training.training_time}s</td>
                    </tr>
                    <tr>
                      <td>Iterations</td>
                      <td>{pipelineMetrics.model_training.iterations}</td>
                    </tr>
                    <tr>
                      <td>Convergence Status</td>
                      <td>{pipelineMetrics.model_training.converged ? 'Converged' : 'Not Converged'}</td>
                    </tr>
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="text-center p-5">
        <Spinner animation="border" role="status" size="lg" />
        <p className="mt-3">Loading pipeline data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="danger" className="m-3">
        {error}
      </Alert>
    );
  }

  return (
    <Container className="py-4">
      <Row className="mb-4">
        <Col>
          <h2 className="mb-3">ML Pipeline Visualization</h2>
          <p className="text-muted">
            Monitor and analyze your machine learning pipeline performance
          </p>
        </Col>
      </Row>

      <Tab.Container id="pipeline-tabs" defaultActiveKey="overview">
        <Row className="mb-3">
          <Col>
            <Nav variant="tabs">
              <Nav.Item>
                <Nav.Link eventKey="overview">Pipeline Overview</Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="status">Runs & Status</Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="metrics">Performance Metrics</Nav.Link>
              </Nav.Item>
            </Nav>
          </Col>
        </Row>
        <Row>
          <Col>
            <Tab.Content>
              <Tab.Pane eventKey="overview">
                <Card className="shadow-sm">
                  <Card.Header className="bg-primary text-white">
                    <h5 className="mb-0">Pipeline Architecture</h5>
                  </Card.Header>
                  <Card.Body>
                    {renderPipelineStages()}
                  </Card.Body>
                </Card>
              </Tab.Pane>
              <Tab.Pane eventKey="status">
                <Card className="shadow-sm">
                  <Card.Header className="bg-primary text-white">
                    <h5 className="mb-0">Pipeline Execution History</h5>
                  </Card.Header>
                  <Card.Body>
                    {renderPipelineStatus()}
                  </Card.Body>
                </Card>
              </Tab.Pane>
              <Tab.Pane eventKey="metrics">
                <Card className="shadow-sm">
                  <Card.Header className="bg-primary text-white">
                    <h5 className="mb-0">Pipeline Performance Metrics</h5>
                  </Card.Header>
                  <Card.Body>
                    {renderMetrics()}
                  </Card.Body>
                </Card>
              </Tab.Pane>
            </Tab.Content>
          </Col>
        </Row>
      </Tab.Container>
    </Container>
  );
}

export default MLPipelineVisualization;