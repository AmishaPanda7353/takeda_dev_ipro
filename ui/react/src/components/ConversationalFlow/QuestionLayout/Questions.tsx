import React from 'react'
import moment from "moment"
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import QuestionLayout from './QuestionLayout'
import { ChatHistoryObject } from '../types';



function Questions({ text, chatObject }: { text: string, chatObject: ChatHistoryObject }) {
    return (
        <QuestionLayout>
            <div className='flex justify-between items-center pl-[80px] pt-[10px]'>
                <div className='flex items-center w-[85%]'>
                    <div><AccountCircleIcon sx={{ color: "#003668", fontSize: 30 }} /></div>
                    <div className='pl-[10px] pr-[10px] text-[14px] font-poppins'>{text}</div></div>
                <div className='w-[15%] pr-[10px] text-[#939393] text-xs'>{moment(chatObject.created_time).format("MMM DD, hh:mm A")}</div>
            </div>
            {/* <DottedLine /> */}
            {/* <InsightNumber num={33} /> */}

        </QuestionLayout>
    )
}

export default Questions