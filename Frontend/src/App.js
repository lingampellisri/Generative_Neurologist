import './App.css';
import Login from './components/login';
import Signup from './components/signup';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Otp from './components/otp';
import VerifyOtp from './components/verifyotp';
import HomePage from './components/homepage';


function App() {
  return (
    <div className="App">
       
    
       <Router>
        <Routes>
        <Route path="/" element={<HomePage/>}/>
          <Route path="/login" element={<Login/>}/>
          <Route path="/signUp" element={<Signup/>}/>
          <Route path="/verify" element={<Otp/>}/>
          <Route path="/verify-otp" element={<VerifyOtp/>}/>
          
        </Routes>
      </Router>
    </div>
  );
}

export default App;
