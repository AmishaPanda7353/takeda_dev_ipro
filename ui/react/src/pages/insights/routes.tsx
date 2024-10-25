import { lazy } from "react";
import { RouteObject } from "react-router-dom";
import Loadable from "../../components/Loadable";

const Insights = Loadable(lazy(() => import(".")));
const Register = Loadable(lazy(() => import("./Register")));


const routes: RouteObject[] = [
    {
        path: "/insights",
        element: <Insights />,
    },
    {
        path: "/register",
        element: <Register />,
    }
];

export default routes;
