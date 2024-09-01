import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import 'bootstrap/dist/css/bootstrap.min.css';


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <div>
    <h1 style={{
      fontSize: '2.5rem',
      color: '#0a74da',
      textAlign: 'left !important',  // Enforces left alignment
      marginBottom: '30px',
      marginLeft: '20px',
      fontWeight: 'bold',
      textShadow: '2px 2px 4px rgba(0, 0, 0, 0.2)'
    }}>
    Generative Neurologist 🧠
  </h1>
  <React.StrictMode>
    <App />
  </React.StrictMode>
  </div>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
