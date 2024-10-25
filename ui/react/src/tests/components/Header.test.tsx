import { cleanup, fireEvent, screen } from "@testing-library/react";
import Header from "components/Header";
import renderWithProviders from "tests/utils";
import Settings from "../../assets/images/settingsicon.png";
import SettingSlider from "components/SettingSlider";
import { Popover } from "@mui/material";
import { useDispatch } from "react-redux";
import { logoutUser } from "store/slice/authSlice";
import { setUser } from "store/slice/userSlice";

afterEach(cleanup);
const settingSliderShow = true;
const setSettingSliderShow = jest.fn();
const open = true;


describe("Header",()=>{
    
jest.mock("react-redux", () => ({
    useDispatch: jest.fn(),
  }));

  jest.mock('store/slice/authSlice')
  jest.mock('store/slice/userSlice')
  
  
  let mockDispatchLogOut: any,mockDispatchSetUser :any;
    beforeEach(() => {
            mockDispatchLogOut = jest.fn();
            mockDispatchSetUser = jest.fn();
      });
      
      test("renders the header", () => {
        renderWithProviders(<Header />);
        expect(screen.getByRole("img", { name: "Insights Pro" })).toBeInTheDocument;
        const SettingDiv = screen.getByTestId("SettingDiv");
        fireEvent.click(SettingDiv);
        expect(screen.getByRole("img", { name: "Settings" })).toBeInTheDocument;
        //  const logout = screen.getByTestId("LogoutDiv")
        // fireEvent.click(logout)
        // fireEvent.click(screen.getByTestId("LogoutDiv"));
        // mockDispatchLogOut({"payload": undefined, "type": "auth/logoutUser"});
        // mockDispatchSetUser({"payload": false,"type": "user/setUser"});
      
        expect(mockDispatchSetUser).toHaveBeenCalledTimes;
        expect(mockDispatchLogOut).toHaveBeenCalledTimes;
        // expect(mockDispatchLogOut).toHaveBeenCalledWith({"payload": undefined, "type": "auth/logoutUser"});
        // expect(mockDispatchSetUser).toHaveBeenCalledWith({"payload": false,"type": "user/setUser"});

    expect(logoutUser).toHaveBeenCalledTimes;
    expect(setUser).toHaveBeenCalledTimes;
    // expect(setUser).toHaveBeenCalledWith(false);
      });

      
      test("rendering SettlingSlider", () => {
        settingSliderShow &&
          renderWithProviders(
            <SettingSlider setSettingSliderShow={setSettingSliderShow} />
          );
      });
      
      const setAnchorEl = jest.fn();
      const setOpen = jest.fn();
      const handleClose = () => {
        setAnchorEl(null);
        setOpen(false);
      };
      test("Testing the PopOver", () => {
        open && renderWithProviders(<Popover open={open} onClose={handleClose} />);
      });
})


// test('toggle setting slider visibility on click', () => {
//     // Render the component
//     renderWithProviders(
//         <div>
//             <div className="cursor-pointer" onClick={() => { setSettingSliderShow(true) }}>
//                 <img src={Settings} alt="Settings" />
//             </div>
//             {settingSliderShow && <SettingSlider setSettingSliderShow={setSettingSliderShow} />}
//         </div>
//     );



// const toggleButton = screen.getByAltText(/settings/i);
// fireEvent.click(toggleButton);



// fireEvent.click(toggleButton);

// });
