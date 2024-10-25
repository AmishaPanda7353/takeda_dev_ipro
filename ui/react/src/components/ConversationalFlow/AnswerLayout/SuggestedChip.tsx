import React from 'react'

function SuggestedChip({ text }: { text: string }) {
    return (
        <div className='font-poppins mt-[25px] mb-[10px] cursor-pointer'>
            <span className='mr-[10px] text-[14px] py-[8px] px-[16px] text-[#18749c] text-center h-[30px] bg-[#F4F8FD] rounded-[23px]'>{text}</span>
        </div>
    )
}

export default SuggestedChip