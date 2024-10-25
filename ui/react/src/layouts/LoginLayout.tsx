import { ThemeProvider } from "@emotion/react";
import { CssBaseline, StyledEngineProvider } from "@mui/material";
import { useAppSelector } from "hooks/store-hooks";
import useGetTheme from "hooks/useGetTheme";
import { Outlet } from "react-router-dom";

function LoginLayout() {
    const mode = useAppSelector((state) => state.theme.mode);
    const theme = useGetTheme(mode);

    return (
        <StyledEngineProvider injectFirst>
            <ThemeProvider theme={theme}>
                <CssBaseline />
                <section className="flex">
                    <div className="w-[100%] h-[100vh] scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-white">
                        <div className="w-[100%] bg-gray-100">
                            <Outlet />
                        </div>
                    </div>
                </section>
            </ThemeProvider>
        </StyledEngineProvider>
    );
}

export default LoginLayout;
