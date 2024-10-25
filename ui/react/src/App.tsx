import React from "react";
import { HashRouter as Router, useRoutes } from "react-router-dom";
import { useAppDispatch, useAppSelector } from 'hooks/store-hooks'
import { loginUser, LoginAPIResponse } from 'store/slice/authSlice';
import { setInitialUserID } from "store/slice/userSlice";
import routes, { loginRts } from "./constants/routes.config";

const GlobalRoutes = () => useRoutes(routes);
const LoginRoutes = () => useRoutes(loginRts);

function App() {
  const dispatch = useAppDispatch()
  const isAuthenticated = useAppSelector((s) => s?.auth?.isAuthenticated);

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
    }
  }, []);

  return (
    <Router>
      {
        isAuthenticated ? (
          <GlobalRoutes />
        ) : (
          <LoginRoutes />
        )
      }
    </Router>
  );
}

export default App;
