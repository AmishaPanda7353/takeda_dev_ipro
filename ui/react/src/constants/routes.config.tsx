
import { RouteObject, Navigate } from "react-router-dom";
import loginRoutes from "pages/login/login.routes";
import LoginCheck from "pages/login/LoginCheck";
import LoginLayout from "layouts/LoginLayout";
import chatbot from "pages/insights/routes"
import Layout from "layouts/Layout";
import ChartLayout from "components/charts/ChartLayout";


const routes: RouteObject[] = [
  {
    element: <LoginLayout />,
    children: [...loginRoutes],
  },
  {
    path: '*',
    element: <Navigate to="/" />,
  },

  {
    element: <Layout />,
    children: [...chatbot]
  },
  {
    path: 'charts',
    element: <ChartLayout />,
  },

];

export const loginRts: RouteObject[] = [
  {
    element: <LoginLayout />,
    children: [...loginRoutes],
  },
  {
    path: '*',
    element: <LoginCheck />,
  }
]

export default routes;