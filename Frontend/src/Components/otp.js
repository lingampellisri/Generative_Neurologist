import React, { useState } from 'react';
import axios from 'axios';
import "./otp.css"
import VerifyOtp from './verifyotp';

function Otp() {
    const [email, setEmail] = useState('');
    const [message, setMessage] = useState('');
    const [otpSent, setOtpSent] = useState(false);

    const sendOtp = async () => {
        try {
            const response = await axios.post(process.env.REACT_APP_BACKEND_URL+'send-otp', { email });
            setMessage(response.data.message);
            setOtpSent(true);
        } catch (error) {
            setMessage('Failed to send OTP');
        }
    };

    if (otpSent) {
        return <VerifyOtp email={email} />;
    }

    return (
        <div className="container">
        <h1 className="heading">Otp Verification</h1>
        <p className="instructionText">
            Enter email to send one time password..!
        </p>
        <input 
            type="email" 
            value={email} 
            onChange={(e) => setEmail(e.target.value)}
            className="emailInput"
            placeholder="Enter your email"
        />
        <button className="continueButton" onClick={sendOtp}>
            Continue
        </button>
        </div>
    );
}

export default Otp;