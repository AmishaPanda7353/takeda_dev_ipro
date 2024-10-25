import { Theme } from "@mui/material";
import darkTheme from "theme/dark";
import lightTheme from "theme/light";

type ThemeList = {
  name: "light" | "dark";
  component: Theme;
};

const themes: ThemeList[] = [
  {
    name: "light",
    component: lightTheme,
  },
  {
    name: "dark",
    component: darkTheme,
  },
];

const useGetTheme = (mode: string): Theme => {
  const selecedTheme = themes.find((item) => item.name === mode)?.component;
  if (selecedTheme) {
    return selecedTheme;
  }
  return lightTheme;
};

export default useGetTheme;
