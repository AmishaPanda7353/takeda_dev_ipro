import { Box, Grid, LinearProgress } from "@mui/material";
// import Logo from "assets/login/Vit_white.png";

function Loading() {
  return (
    <Grid
      data-testid="loading-component"
      container
      spacing={2}
      direction="column"
      alignItems="center"
      justifyContent="center"
      style={{ minHeight: "85vh" }}
    >
      <Grid xs={12} sm={12} md={12} lg={12} item>
        {/* <Box
          className="w-24 animate-bounce"
          component="img"
          // src={Logo}
          alt="Logo"
        /> */}
      </Grid>
      <Grid xs={12} sm={12} md={12} lg={12} item>
        <LinearProgress data-testid='loader' className="w-60" />
      </Grid>
    </Grid>
  );
}

export default Loading;
