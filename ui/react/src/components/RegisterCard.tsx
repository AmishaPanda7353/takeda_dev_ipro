import { useNavigate } from "react-router-dom";
import _ from "lodash";
import { useMsal } from '@azure/msal-react'
import { useAppDispatch } from 'hooks/store-hooks';
import { setInitialUserID } from "store/slice/userSlice";
import { SetStateAction, useEffect, useState } from "react";
import { useAppSelector } from "store";
import axios from "axios";
import OutlinedInput from '@mui/material/OutlinedInput';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import ListItemText from '@mui/material/ListItemText';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import Checkbox from '@mui/material/Checkbox';
import KeyboardBackspaceIcon from '@mui/icons-material/KeyboardBackspace';

import { loginRequest } from '../auth.config';
import { UserObject, loginUser, LoginAPIResponse } from '../store/slice/authSlice';
import InsightsPro from '../assets/images/Insightsprologo.png'



const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};


function RegisterCard() {
  const isAuthenticated  = false;
  // const { isAuthenticated } = useAppSelector((s) => s.auth);
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const { instance } = useMsal()

  const defaultExpiryDate = `${(new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)).toISOString().split('T')[0]}`

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [selectedOption, setSelectedOption] = useState('');
  const [expiryDate, setExpiryDate] = useState(defaultExpiryDate);
  const [showExpiryDateInput, setShowExpiryDateInput] = useState(false);
  const [error, setError] = useState('');

  const [selectedDomainName, setSelectedDomainName] = useState<string[]>([]);

  const userId = JSON.parse(localStorage.getItem('auth_state') as string).userInfo.name;

  const routechange = () => {
    const path = "/login"
    navigate(path)
  }

  const routeInsights = () => {
    const path = "/insights"
    navigate(path)
  };

  const routechangeNoAccess = () => {
    const path = "/nopermission"
    navigate(path)
  }

  const formatDateForInput = (date: string) => {
    const [day, month, year] = date.split('-');
    return `${year}-${month}-${day}`;
  };

  
  const handleOptionChange = (value: string) => {
    setSelectedOption(value);

    if (value === 'external') {
      setShowExpiryDateInput(true);
    } else {
      setShowExpiryDateInput(false);
      const currentDate = new Date();

      const futureDate = new Date(currentDate);
      futureDate.setDate(currentDate.getDate() + 50);
    
      const formattedDate = `${futureDate.getFullYear()}-${
        (futureDate.getMonth() + 1).toString().padStart(2, '0')
      }-${futureDate.getDate().toString().padStart(2, '0')}`;
      setExpiryDate(formattedDate)
    }
  };

  const handleRegister = async () => {
    try {
      
      if(selectedOption === 'external' && expiryDate !== defaultExpiryDate)
        setExpiryDate(formatDateForInput(expiryDate))

      // console.log(username, password, confirmPassword, selectedOption, expiryDate, selectedDomainName)


      if(password === confirmPassword){
        const response = await axios.post(`${process.env.REACT_APP_API_URL}api/create_user`, {
          user_id   : username,
          pwd      : password,
          usertype : selectedOption,
          domain   : selectedDomainName.join(),
          exp_date : expiryDate
        });
        if(response.data.response === 'User Already Exists!!!')
            setError('User Already Exists!!!')
        else
            routechange()
      }
      else{
        setError('Passwords do not match.')
      }

    } catch (error) {
      setError('API failed.')
    }
  }
  const [domains, setDomains] = useState<string[]>([]);
  
  useEffect(() => {

    const apiUrl = `${process.env.REACT_APP_API_URL}api/get_data_names`;
    const postData = { user_id: userId }; 

    axios
      .post(apiUrl, postData)
      .then((response) => {
        setDomains(['All', ...response.data.data_names]);
      })
      .catch((error) => {
        console.error('Error fetching domains:', error);
      });

  }, [])


  const handleDomainChange = (event: SelectChangeEvent<typeof selectedDomainName>) => {
    const { value } = event.target;
    if (value.includes("All")) {
      setSelectedDomainName(domains.filter(domain => domain !== "All"));
    } else {
      setSelectedDomainName(typeof value === 'string' ? value.split(',') : value,);
    }
  };

  const handleSubmit = (e: { preventDefault: () => void; }) => {
    e.preventDefault();
    console.log('Form submitted:');
    handleRegister();
  };

  return (
    <div className="flex py-5 px-20 flex-col justify-start rounded-[20px] min-h-[400px] min-w-[400px] bg-white">
      <div className="flex mt-10 justify-center text-lg">
        <KeyboardBackspaceIcon className="cursor-pointer" onClick={routeInsights} />
        <h1 className="text-center flex-grow">Register</h1>
      </div>

      <div className="justify-center align-end content-center">
      <form onSubmit={handleSubmit}>
        <div className="relative">
            <span className="absolute top-[25%] left-3 text-gray-600">
              <i className="material-icons">person</i>
            </span>
            <input
              type="text"
              placeholder="Username"
              className="w-full py-2 px-10 my-2 rounded border border-gray-300 focus:outline-none focus:border-[#18749C] text-[14px] font-poppins"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>

          <div className="relative">
            <span className="absolute top-[25%] left-3 text-gray-600">
              <i className="material-icons">lock</i>
            </span>
            <input
              type="password"
              placeholder="Password"
              className="w-full py-2 px-10 my-2 rounded border border-gray-300 focus:outline-none focus:border-[#18749C] text-[14px] font-poppins"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <div className="relative">
            <span className="absolute top-[25%] left-3 text-gray-600">
              <i className="material-icons">lock</i>
            </span>
            <input
              type="password"
              placeholder="Confirm Password"
              className="w-full py-2 px-10 my-2 rounded border border-gray-300 focus:outline-none focus:border-[#18749C] text-[14px] font-poppins"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
          </div>

          <div>
            <FormControl sx={{ width: 300 }}>
              <InputLabel id="demo-multiple-checkbox-label">Domain</InputLabel>
              <Select
                labelId="demo-multiple-checkbox-label"
                id="demo-multiple-checkbox"
                multiple
                value={selectedDomainName}
                onChange={handleDomainChange}
                input={<OutlinedInput label="Domain" />}
                renderValue={(selected) => selected.join(', ')}
                MenuProps={MenuProps}
                style={{ fontFamily: 'sans-serif', width: '100%', height: 'auto'}}
              >
                {domains.map((domain) => (
                  <MenuItem key={domain} value={domain} disabled={selectedDomainName.includes("All") && domain !== "All"}>
                    <Checkbox checked={selectedDomainName.indexOf(domain) > -1} />
                    <ListItemText primary={domain} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </div>

          <div className="flex flex-col mt-2">
          <label>
            <input
              type="radio"
              name="options"
              value="internal"
              checked={selectedOption === 'internal'}
              onChange={() => handleOptionChange('internal')}
            />
            Internal
          </label>
          <label>
            <input
              type="radio"
              name="options"
              value="external"
              checked={selectedOption === 'external'}
              onChange={() => handleOptionChange('external')}
            />
            External
          </label>
          </div>

          {showExpiryDateInput && (
          <div className="relative">
            <label htmlFor="expiryDate" className="block text-gray-600 text-sm font-semibold mb-1">
              Expiry Date
            </label>
            <input
              type="date"
              placeholder="Expiry Date"
              className="w-full py-2 px-10 my-2 rounded border border-gray-300 focus:outline-none focus:border-[#18749C] text-[14px] font-poppins"
              defaultValue={`${(new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)).toISOString().split('T')[0]}`}
              onChange={(e) => setExpiryDate(e.target.value)}
              min={expiryDate}
            />
          </div>
        )}

      {error && <div className="text-red-500 text-sm mt-2">{error}</div>}


          <button
            data-testid="routechange-button"
            type="button"
            className="w-full bg-[#18749C] text-[14px] font-poppins text-white font-medium py-2 mt-4 rounded"
            onClick={handleRegister}
          >
            Register
          </button>
        
          <button type="submit" style={{display: 'none'}}>Submit</button>

        </form>
      </div>

      <div className="flex justify-center text-[12px] text-[#9B9898] mt-5">
        Powered by Tiger Analytics
      </div>
    </div>

  )
}

export default RegisterCard;