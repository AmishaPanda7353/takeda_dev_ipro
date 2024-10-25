/* eslint-disable security/detect-object-injection */
/* eslint-disable camelcase */

import React, { useEffect, useState } from 'react';
import { useAppSelector, useAppDispatch } from "store";
import { ENDPOINTS } from 'constants/constant';
import client from 'client';
import { setOutletLoading, setSelectedDomain } from 'store/slice/userSlice';
import BouncingDotsLoader from 'pages/insights/BouncingDotsLoader';
import SlideOver from './SlideOver';



function DropDownComponent({ setDropDown, onApply }: { setDropDown: (flag: boolean) => void, onApply: (value: string) => void }) {
    const dispatch = useAppDispatch();
    const { user_id, selectedDomain, outletLoading } = useAppSelector((s) => s.user);
    const [localSelectedDomain, setLocalSelectedDomain] = useState(selectedDomain);
    const [domainList, setDomainList] = useState([]);

    const handleDropChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        setLocalSelectedDomain(e.target.value);
    }

    const handleApplyClick = () => {
        dispatch(setSelectedDomain(localSelectedDomain));
        onApply(localSelectedDomain);
        setDropDown(false);
    }

    const getDataFromAPI: any = async () => {
        dispatch(setOutletLoading(true));
        try {
            const response = await client.post(ENDPOINTS.dataNames, { user_id });
            const resValue = response.data;
            if (resValue && resValue.data_names && resValue.data_names.length > 0) {
                const initialValue = resValue.data_names[0];
                setLocalSelectedDomain(prevValue => prevValue || initialValue); // Use localSelectedDomain if it's already set, otherwise use initialValue
            }
            return resValue;
        } catch (error) {
            console.error("Error fetching data :", error);
            dispatch(setOutletLoading(false));
            return null;
        }
        finally {
            dispatch(setOutletLoading(false));
        }
    }

    useEffect(() => {
        getDataFromAPI()
            .then((data: any) => {
                if (data && data.data_names) {
                    setDomainList(data.data_names);
                }
            });
    }, []);

    return (
        <div>
            <SlideOver closeIcon={!!selectedDomain} closeOnClick={() => setDropDown(false)}>
                {outletLoading ? (
                    // Loading component
                    <div className='flex justify-center items-center h-screen'>
                    <div className='flex justify-around items-center'>
                      <span className='font-bold text-sm mr-[10px]'>Loading</span>
                      <BouncingDotsLoader />
                    </div>
                  </div>
                  
                ) :
                    (<div>
                        <div className='py-4'>
                            <h3 className='mb-2'>Select Domain</h3>
                        </div>
                        <select onChange={handleDropChange} value={localSelectedDomain} className='border rounded-lg py-2 px-3 w-[80%] focus:outline-none  focus:border-blue-300'>
                            {domainList.map((domain) => (
                                <option className="text-[16px]" key={domain} value={domain}>
                                    {domain}
                                </option>
                            ))}
                        </select>
                        <div className="mt-6">
                            <button type='button' className="bg-blue-600 text-white font-semibold py-2 px-4 rounded shadow-md transition duration-300 ease-in-out" onClick={handleApplyClick}>Apply</button></div>
                    </div>)}
            </SlideOver>
        </div>
    )
}
export default DropDownComponent;
