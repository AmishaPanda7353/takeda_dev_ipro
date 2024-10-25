import React from "react";
import { useNavigate } from "react-router-dom";
import { useAppDispatch } from 'hooks/store-hooks'
import { loginUser, LoginAPIResponse } from 'store/slice/authSlice';
import { setInitialUserID } from "store/slice/userSlice";

function LoginCheck() {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  React.useEffect(() => {
    const user = sessionStorage.getItem("user_info");
    const token = sessionStorage.getItem("access_token");
    const userGroups = sessionStorage.getItem("user_groups");
    if (user && token && userGroups) {
      const loginResp: LoginAPIResponse = {
        user_info: JSON.parse(user) || {},
        token: sessionStorage.getItem("access_token") || '',
        groups: JSON.parse(userGroups) || [],
      }
      dispatch(loginUser(loginResp));
      dispatch(setInitialUserID(loginResp?.user_info?.email ? loginResp?.user_info?.email : ''));
    } else {
      // sessionStorage.removeItem("user_info");
      // sessionStorage.removeItem("access_token");
      // sessionStorage.removeItem("user_groups");
      sessionStorage.clear();
      navigate("/");
    }
  }, []);
  return (
    <>
    </>
  )
}

export default LoginCheck;
