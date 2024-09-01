import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import React from "react";
import { useState } from "react";
import { BiSolidLockAlt } from "react-icons/bi";
import { FaUser } from "react-icons/fa";
// import { Link } from 'react-router-dom';
import "./login.css"

const Login = () => {
  let [name, setUsername] = useState(''); // State for username
  let [password, setPassword] = useState(''); // State for password
  // let [email, setEmail] = useState('');
  let [invalid, setInvalidCredentials] = useState("");

  let navigate = useNavigate();

  const uHandler = (e) => {
    setUsername(e.target.value);
  }

  const pwHandler = (e) => {
    setPassword(e.target.value);
  }

  const submitHandler = async (e) => {
    e.preventDefault();
    console.log(name, password);
    try {
      let response = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/login`, {
        name: name,
        password: password
      },
        {
          auth: {
            name: name,
            password: password
          }
        });

      console.log(response);

      if (response.status === 200 || response.status == 201 && response.data.length > 0) {
        localStorage.setItem("loggedIn", true);
        localStorage.setItem("rname", response.data.name);
        localStorage.setItem("rphone", response.data.phone);
        console.log("Login successful");
        navigate('/');
        // res.send(user);
        // Redirect to the home page or desired route upon successful login
      } else {
        console.log("Invalid credentials");
        setInvalidCredentials("Incorrect username or password!!");
      }
    } catch (error) {
      console.log("Error occurred during login:", error);
      setInvalidCredentials("Incorrect username or password!!");
    }

  }


  return (
    // <div className="b1">
    //   <div className='box'>
    //     <form>
    //       <h1>Login!</h1>
    //       <div className='input'>
    //         <input type="text" placeholder='Username' onChange={uHandler} />
    //         <FaUser className="icon" />
    //       </div>

    //       <div className='input'>
    //         <input type="password" placeholder='Password' onChange={pwHandler} />
    //         <BiSolidLockAlt className="icon" />
    //       </div>
    //       <div className='remember'>
    //         <label><input type="checkbox" />Remember me</label>
    //         <a href="#">Forgot password</a>
    //       </div>
    //       <button type="submit" onClick={submitHandler}>Login</button>
    //       {/* <Link to="/verify-otp" className="continueButton" onClick={submitHandler}>
    //     Login
    //     </Link> */}

    //       <div className='register'>
    //         <p>Don't have an account? <a href="/signUp">Register</a></p>
    //       </div>
    //       {invalid && <div className="error-message">{invalid}</div>}
    //     </form>

    //   </div>
    // </div>
     <div className="min-h-screen flex items-center justify-center bg-blue-100 p-4">
      <div className="w-full max-w-md bg-white p-6 rounded-lg shadow-md">
        <form>
          <h1 className="text-2xl sm:text-3xl font-bold mb-6 text-center">Login!</h1>
          <div className="mb-4">
            <div className="flex items-center border-b border-gray-300 py-2">
              <FaUser className="text-gray-400 mr-3" />
              <input
                type="text"
                placeholder="Username"
                className="w-full py-2 px-3 text-gray-700 focus:outline-none"
                onChange={uHandler}
              />
            </div>
          </div>
          <div className="mb-4">
            <div className="flex items-center border-b border-gray-300 py-2">
              <BiSolidLockAlt className="text-gray-400 mr-3" />
              <input
                type="password"
                placeholder="Password"
                className="w-full py-2 px-3 text-gray-700 focus:outline-none"
                onChange={pwHandler}
              />
            </div>
          </div>
          <div className="flex items-center justify-between mb-6">
            <label className="flex items-center">
              <input type="checkbox" className="mr-2" />
              Remember me
            </label>
            <a href="#" className="text-blue-500 hover:text-blue-700">Forgot password</a>
          </div>
          <button
            type="submit"
            className="w-full bg-gray hover:bg-black-700 text-black font-bold py-2 px-4 rounded"
            onClick={submitHandler}
          >
            Login
          </button>
          {/* <Link to="/verify-otp" className="continueButton" onClick={submitHandler}>
            Login
            </Link> */}
          <div className="mt-6 text-center">
            <p>
              Don't have an account? <a href="/signUp" className="text-blue-500 hover:text-blue-700">Register</a>
            </p>
          </div>
          {invalid && <div className="mt-4 text-red-500 text-center">{invalid}</div>}
        </form>
      </div>
    </div>
     
  );
}

export default Login;
