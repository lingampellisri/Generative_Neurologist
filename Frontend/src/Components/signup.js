import React, { useEffect, useState } from "react"
import axios from "axios"
// import { useNavigate, Link } from "react-router-dom"
import { BiLockAlt as BiSolidLockAlt } from "react-icons/bi";
import { FaUser, FaPhone } from "react-icons/fa";
import { MdEmail } from "react-icons/md";
import { Link } from 'react-router-dom';
// import "./login.css";


const Signup = () => {
  const [formObj, setFormObj] = useState({ name: "", password: "", email: "", phone: "" });
  let [signedUp, setSignedUp] = useState(false);
  let [errorSigningUp, setErrorSigningUp] = useState('');
  const [invalidCredentials, setInvalidCredentials] = useState(false);
  const [invalidCredMsg, setInvalidCredMsg] = useState("");
  const changeHandler = (e) => {
    //console.log(formObj);
    setFormObj({ ...formObj, [e.target.name]: e.target.value });
  };

  const [email, setEmail] = useState('');
  const [userEnteredOtp, setUserEnteredOtp] = useState('');
  const [message, setMessage] = useState('');

  const sendOtp = async () => {
    try {
      const response = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/send-otp`, { email });
      setMessage(response.data.message);
    } catch (error) {
      setMessage('Failed to send OTP');
    }
  };


  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log(formObj);

    // submit it to the backend server
    try {

      let resp = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/signUp`, { ...formObj });
      console.log(resp);
      if (resp.data) {
        setSignedUp(true);
        console.log(" successfully signedup");
      }
      else {
        setSignedUp(false);
        setErrorSigningUp(" Error while signing up");
      }

    }
    catch (error) {
      console.log(" error while signing");
      console.log(error);
      setSignedUp(false);
      setErrorSigningUp(" Error while signing up");

    }
  };

  return (
    signedUp ? (
      <div className="min-h-screen flex items-center justify-center bg-blue-100 p-4">
        <div className="registration-container">
          <h1 className="text-2xl sm:text-3xl font-bold mb-6 text-center">Otp Verification</h1>
          <p className="text-center mb-4">Enter email to send one time password..!</p>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full py-2 px-3 text-gray-700 border border-gray-300 rounded focus:outline-none"
            placeholder="Enter your email"
          />
          <Link
            to="/verify-otp"
            className="block bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 mt-4 rounded text-center"
            onClick={sendOtp}
          >
            Continue
          </Link>
        </div>
      </div>
    ) : (
      <div className="min-h-screen flex items-center justify-center bg-blue-100 p-4">
        <div className="b1">
          <div className='box'>
            <form onSubmit={handleSubmit}>
              <h1 className="text-2xl sm:text-3xl font-bold mb-6 text-center">Register</h1>
              {invalidCredentials && <div className="error">{invalidCredMsg}</div>}
              <div className='input'>
                <input
                  type="text"
                  name="username"
                  id="username"
                  placeholder='Username'
                  onChange={changeHandler}
                  className="w-full py-2 px-3 text-gray-700 border border-gray-300 rounded focus:outline-none"
                />
                <FaUser className="icon" />
              </div>
              <div className='input'>
                <input
                  type="password"
                  name="password"
                  id="password"
                  placeholder='Password'
                  onChange={changeHandler}
                  className="w-full py-2 px-3 text-gray-700 border border-gray-300 rounded focus:outline-none"
                />
                <BiSolidLockAlt className="icon" />
              </div>
              <div className='input'>
                <input
                  type="email"
                  name="email"
                  id="email"
                  placeholder="Email"
                  required
                  onChange={changeHandler}
                  className="w-full py-2 px-3 text-gray-700 border border-gray-300 rounded focus:outline-none"
                />
                <MdEmail className="icon" />
              </div>
              <div className='input'>
                <input
                  type="text"
                  name="phone"
                  email="phone"
                  placeholder="Phone No"
                  required
                  onChange={changeHandler}
                  className="w-full py-2 px-3 text-gray-700 border border-gray-300 rounded focus:outline-none"
                />
                <FaPhone className="icon" />
              </div>
              <button
                type="submit"
                className="w-full bg-gray hover:bg-black-700 text-black font-bold py-2 px-4 rounded"
              >
                Register
              </button>
              <div className='register'>
                <p><a href="/login" className="text-black hover:text-black">Login</a></p>
              </div>
            </form>
          </div>
        </div>
      </div>
    )
  );
};


export default Signup;
