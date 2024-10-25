import { useNavigate } from "react-router-dom";
import _ from "lodash";
import { useMsal } from '@azure/msal-react'
import { useAppDispatch } from 'hooks/store-hooks';
import { setInitialUserID } from "store/slice/userSlice";
import { useState, useEffect } from "react";
import { useAppSelector } from "store";
import axios from "axios";
import { loginRequest } from '../auth.config';
import { UserObject, loginUser, LoginAPIResponse } from '../store/slice/authSlice';
import InsightsPro from '../assets/images/Insightsprologo.png'

function LoginCard() {
  let isAuthenticated  = false;
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const { instance } = useMsal()

  const [username, setUsername] = useState('');
  const [passwordd, setPassword] = useState('');
  const [error, setError] = useState(' ');

  const storedAuthData = JSON.parse(localStorage.getItem('auth_state') as string)

  const routechange = () => {
    const path = "/insights"
    navigate(path)
  }

  const routechangeNoAccess = () => {
    const path = "/nopermission"
    navigate(path)
  }

  const handleLogin = async () => {
    // instance.loginRedirect(loginRequest)
    
    try {
        const loginResponse = await instance.loginPopup();
        isAuthenticated = true;
        const user: UserObject = {
          id: 1,
          name: loginResponse.account?.name || "",
          email: loginResponse.account?.username || "",
          picture: 'na',
          last_login: 'na',
          date_joined: 'na',
        };
        const loginRespSend: LoginAPIResponse = {
          user_info: user,
          token: loginResponse.accessToken,
          groups: [],
        }
        dispatch(loginUser(loginRespSend));
        dispatch(setInitialUserID(loginRespSend?.user_info?.email ? loginRespSend?.user_info?.email : ''));
        routechange();

        localStorage.setItem('lock', 'true')
        console.log('Login success', loginResponse);
      } catch (err) {
        const error = String(err)
        if(error.includes('Error'))
          setError(error.substring(6))
        else
          setError('Invalid credentials. Please try again.');
      }
  }

  const handleLogin2 = async () => {
    console.log('clicked..', username, passwordd)

    try {
      console.log('inside try..', username, passwordd)

      if(localStorage.getItem('lock') === 'true')
        throw Object.assign(new Error('Single user login session will be only allowed at a given time.'));
      
      const response = await axios.post(`${process.env.REACT_APP_API_URL}api/verify_user`, {
        user_id: username, 
        password: passwordd,
      });
      console.log('login...',JSON.parse(response.data.response).message)
      if(response.data.response.includes('User Verified!')){
        if(JSON.parse(response.data.response).user_type === 'admin')
          localStorage.setItem('user_type', 'admin')
        console.log('Login successful', response.data);

        localStorage.setItem('lock', 'true')

        isAuthenticated = true;
        const user: UserObject = {
          id: 1,
          name: username || "",
          email: username || "",
          picture: 'na',
          last_login: 'na',
          date_joined: 'na',
        };
        const loginRespSend: LoginAPIResponse = {
          user_info: user,
          token: '',
          groups: [],
        }
        dispatch(loginUser(loginRespSend));
        dispatch(setInitialUserID(loginRespSend?.user_info?.email ? loginRespSend?.user_info?.email : ''));
        routechange();

      }
      else{
        setError('Invalid credentials. Please try again.');

      }
    } catch (err) {
      const error = String(err)
      if(error.includes('Error'))
        setError(error.substring(6))
      else
        setError('Invalid credentials. Please try again.');
    }

  }

  const redirectLogin = async () => {
    try {
      const responseMSAL = await instance.handleRedirectPromise();
      if (responseMSAL) {
        // console.log(responseMSAL);
        const respUserGroups = await axios.get('https://graph.microsoft.com/v1.0/me/memberOf', {
          headers: {
            Authorization: `Bearer ${responseMSAL.accessToken}`
          }
        });
        const userGroups = respUserGroups.data?.value ? respUserGroups.data?.value : [];
        const accessGrpIx = _.findIndex(userGroups, (i: any) => i.id === "481aa528-3b2a-4477-9577-e911073fa690");
        if ((userGroups.length > 0 && accessGrpIx !== -1)) {
          const user: UserObject = {
            id: 1,
            name: responseMSAL?.account?.name || "",
            email: responseMSAL?.account?.username || "",
            picture: 'na',
            last_login: 'na',
            date_joined: 'na',
          };
          const loginResp: LoginAPIResponse = {
            user_info: user,
            token: responseMSAL?.accessToken,
            groups: userGroups,
          }
          sessionStorage.setItem('access_token', responseMSAL?.accessToken)
          sessionStorage.setItem('user_info', JSON.stringify(user))
          sessionStorage.setItem('user_groups', JSON.stringify(userGroups))
          dispatch(loginUser(loginResp));
          dispatch(setInitialUserID(loginResp?.user_info?.email ? loginResp?.user_info?.email : ''));
          routechange();
        } else {
          routechangeNoAccess();
        }
      } else {
        handleLogin()
      }
    } catch (error) {
      console.log(error);
    }
  }

  useEffect(() => {
    if (storedAuthData && storedAuthData.isAuthenticated) {
      routechange();
    }
  }, [])

  return (
    <div className="flex py-5 px-20 flex-col justify-between rounded-[20px] min-h-[400px] min-w-[400px] bg-white">
  <div className="flex mt-10 justify-center text-lg">
    <img src={InsightsPro} alt="Insights Pro" />
  </div>
  
  <div className="justify-center align-end content-center">
    <input
      type="text"
      placeholder="Username"
      className="w-full py-2 px-4 my-2 rounded border border-gray-300 focus:outline-none focus:border-[#18749C] text-[14px] font-poppins"
      value={username}
      onChange={(e) => setUsername(e.target.value)}
    />
    
    <input
      type="password"
      placeholder="Password"
      className="w-full py-2 px-4 my-2 rounded border border-gray-300 focus:outline-none focus:border-[#18749C] text-[14px] font-poppins"
      value={passwordd}
      onChange={(e) => setPassword(e.target.value)}
    />
    {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
    
    <button
      data-testid="normal-routechange-button"
      type="button"
      className="w-full bg-[#18749C] text-[14px] font-poppins text-white font-medium py-2 mt-4 rounded"
      onClick={() => handleLogin2()}
    >
      Login
    </button>

    <button
      data-testid="routechange-button"
      type="button"
      className="w-full bg-[#18749C] text-[14px] font-poppins text-white font-medium py-2 mt-4 rounded"
      onClick={() => handleLogin()}
    >
      Login with Microsoft
    </button>
  </div>
  
  <div className="flex justify-center text-[12px] text-[#9B9898]">
    Powered by Tiger Analytics
  </div>
</div>

  )
}

export default LoginCard;