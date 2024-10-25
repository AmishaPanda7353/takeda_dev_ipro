import { lazy } from "react";
import { RouteObject } from "react-router-dom";
import Loadable from "../../components/Loadable";

const Login = Loadable(lazy(() => import("./Login")));
const NoPermission = Loadable(lazy(() => import("./NoPermission")));

const loginRoutes: RouteObject[] = [
    {
        path: "/",
        element: <Login />,
    },
    {
        path: "/nopermission",
        element: <NoPermission />,
    },
];

export default loginRoutes;
