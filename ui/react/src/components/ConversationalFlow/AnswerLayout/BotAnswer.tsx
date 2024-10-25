import React from 'react'
import moment from "moment"
import AnswerLayout from './AnswerLayout';
import BotImage from "../../../assets/images/BotImage.svg"
import UserAnswerFeedback from '../UserAnswerFeedback';
import { ChatHistoryObject } from '../types';


function BotAnswer({ text, chatObject }: { text: string, chatObject: ChatHistoryObject }) {
    return (
        <AnswerLayout >
            <div className='flex justify-start items-center pl-[60px] pt-[20px] pb-[20px]'>
                <div className='flex justify-center items-center bg-black rounded-full h-[26px] w-[26px]'>
                    <img className='w-[20px]' src={BotImage} alt="chatbotICon" />
                </div>
                <div className='pl-[10px] pr-[10px] max-w-[80%] text-[14px] font-poppins text-[#1c1c1c]'>{text}</div>
                {/* <div className='pr-[10px] text-[#939393] text-xs'>{moment(chatObject.created_time).format("MMM DD, hh:mm A")}</div> */}
                <UserAnswerFeedback key={String(chatObject.feedback)} answerId={chatObject.answer_id.toString()}
                    defaultLike={chatObject.feedback !== null && chatObject.feedback === true}
                    defaultDislike={chatObject.feedback !== null && chatObject.feedback === false}
                    contextId={chatObject.context_id} />
            </div>
            {/* <div className='flex justify-center flex-wrap'> <SuggestedChip text="Lorem ipsum the artificial intelligence company founded by?" /><SuggestedChip text="Which shipments are delayed?" /><SuggestedChip text="Lorem ipsum the artificial intelligence company founded by?" /><SuggestedChip text="Which shipments are delayed?" /><SuggestedChip text="Lorem ipsum the artificial intelligence company founded by?" /><SuggestedChip text="Which shipments are delayed?" /></div> */}

        </AnswerLayout>
    )
}

export default BotAnswer