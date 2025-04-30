import React, { useState } from 'react';
import { Container, Navbar, Nav } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import App from './App';  // Your existing main component
import MLPipelineVisualization from './MLPipelineVisualization';  // New ML Pipeline component
import './App.css';

function MainAppRouter() {
  const [currentPage, setCurrentPage] = useState('prediction');

  return (
    <>
      <Navbar bg="primary" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand href="#home">Forex Rate Prediction</Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link 
                href="#prediction" 
                active={currentPage === 'prediction'}
                onClick={() => setCurrentPage('prediction')}
              >
                Prediction Tool
              </Nav.Link>
              <Nav.Link 
                href="#pipeline" 
                active={currentPage === 'pipeline'}
                onClick={() => setCurrentPage('pipeline')}
              >
                ML Pipeline
              </Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      {currentPage === 'prediction' ? <App /> : <MLPipelineVisualization />}
    </>
  );
}

export default MainAppRouter;