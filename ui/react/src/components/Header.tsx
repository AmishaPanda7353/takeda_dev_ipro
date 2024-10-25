/* eslint-disable security/detect-object-injection */
/* eslint-disable camelcase */

import React, { useState } from 'react'
import { useMsal } from '@azure/msal-react'
import { setOutletLoading, setSelectedDomain, setUser, setUserChatDetails } from "store/slice/userSlice";
import { logoutUser } from 'store/slice/authSlice';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import LogoutIcon from '@mui/icons-material/Logout';
import HowToRegIcon from '@mui/icons-material/HowToReg';
import Popover from '@mui/material/Popover';
import { useAppDispatch, useAppSelector } from 'hooks/store-hooks'
import { getUserChatDetailsReponse } from 'constants/common';
import { useNavigate } from "react-router-dom";
import { logoutRequest } from '../auth.config'
import SettingSlider from './SettingSlider';
import Settings from '../assets/images/settingsicon.png'
import InsightsPro from '../assets/images/InsightsproHeader.png'
import DropDownComponent from './DropDownComponent';


function Header() {
  const { instance } = useMsal()
  const [open, setOpen] = useState<boolean>(false);
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const [settingSliderShow, setSettingSliderShow] = useState<boolean>(false);
  const [dropDown, setDropDown] = useState<boolean>(true);
  // const [selectedDomain, setSelectedDomain] = useState<string>('');
  const dispatch = useAppDispatch()
  const { user_id, selectedDomain } = useAppSelector((s) => s.user);
  const navigate = useNavigate()

  const logoutroute = () => {
    // localStorage.removeItem('nlp_insights_domain_selection');
    instance.logout(logoutRequest);
    dispatch(logoutUser())
    dispatch(setUser(false));

  }

  const registerroute = () => {
    const path = "/register"
    navigate(path)
  }

  const handleClick = (event: any) => {
    setAnchorEl(event.currentTarget);
    setOpen(true)
  };

  const handleClose = () => {
    setAnchorEl(null);
    setOpen(false)
  };

  // Handler for applying the selected domain

  const handleApplyDomain = async (value: string) => {
    if (value !== selectedDomain) {
      setSelectedDomain(value);
      const payload: any = { user_id, data_name: value };
      dispatch(setOutletLoading(true));
      const response = await getUserChatDetailsReponse(payload);
      console.log(response.data.chat_details.data_name, "res")
      dispatch(setUserChatDetails(response.data));
      dispatch(setOutletLoading(false));
    }
    setDropDown(false); // close the dropdown
  }
  return (
    <div data-testid="outerDiv" className="bg-white px-4 py-3 flex items-center w-full shadow-[0_3px_6px_#00000019]">
      <div className="flex items-center w-[50%]">
        <div className="text-xl tracking-[0px] leading-[30px] text-[#1C1C1C] font-semibold"><img src={InsightsPro} alt='Insights Pro' /></div>
      </div>
      <div className="flex items-center w-[50%] justify-end px-3 pr-5">
        <div className="float-right flex gap-4">
          <div data-testid="SettingDiv"
            className="cursor-pointer"
            onClick={() => { setDropDown(true) }}>

            <img className='' src={Settings} alt="Settings" />
          </div>
          {dropDown &&
            <DropDownComponent setDropDown={setDropDown} onApply={handleApplyDomain} />
            // <SettingSlider setSettingSliderShow={setSettingSliderShow} />
          }
          <div className="cursor-pointer  mt-0.5">
            <AccountCircleIcon sx={{ color: "#003668", fontSize: 26 }} onClick={handleClick} />
            {
              open &&
              <Popover
                open={open}
                anchorEl={anchorEl}
                onClose={handleClose}
                anchorOrigin={{
                  vertical: 40,
                  horizontal: 'left',
                }}
              >
                { localStorage.getItem('user_type') === 'admin' &&
                <div data-testid="LogoutDiv" onClick={registerroute}
                  className="flex p-4 cursor-pointer justify-center align-center items-center">
                  <HowToRegIcon fontSize='small' sx={{ marginRight: 1 }} />
                  <span>Register</span>
                </div>}
                
                <div data-testid="LogoutDiv" onClick={logoutroute}
                  className="flex p-4 text-[#F01F1F] cursor-pointer justify-center align-center items-center">
                  <LogoutIcon fontSize='small' sx={{ marginRight: 1 }} />
                  <span>Logout</span>
                </div>
              </Popover>
            }
          </div>
        </div>
      </div>
    </div>
  )
}


export default Header