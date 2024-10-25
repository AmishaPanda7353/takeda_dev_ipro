import { ThemeProvider } from "@emotion/react";
import { CssBaseline, StyledEngineProvider } from "@mui/material";
import Sidebar from "components/sidebar";
import { MenuItemType } from "components/sidebar/types";
import Header from "components/Header";
import { useAppSelector } from "hooks/store-hooks";
import useGetTheme from "hooks/useGetTheme";
import { Outlet } from "react-router-dom";
import Footer from "components/footer";
import Loading from "components/Loading";



function Layout() {
    const mode = useAppSelector((state) => state.theme.mode);
    const { outletLoading } = useAppSelector((state) => state.user);
    const theme = useGetTheme(mode);

    const menus: MenuItemType[] = [];

    return (
        <StyledEngineProvider injectFirst>
            <ThemeProvider theme={theme}>
                <CssBaseline />
                <section className="flex">
                    <Sidebar menuItems={menus} initialOpenState />
                    <div className="w-[100%] bg-white scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-white">
                        <Header />
                        <div className="pl-5 pr-6 pt-5 pb-1 w-[100%] h-[calc(100vh-85px)]">
                            {
                                outletLoading ? <Loading /> : <Outlet />
                            }
                        </div>
                        <Footer />
                    </div>
                </section>
            </ThemeProvider>
        </StyledEngineProvider>
    );
}

export default Layout;
