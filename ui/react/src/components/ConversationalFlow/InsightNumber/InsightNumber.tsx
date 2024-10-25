import React from 'react'

function InsightNumber({ num }: { num: string }) {
    return (
        <div className='flex text-[#1C1C1C] text-[60px] font-semibold cursor-default'>{num}</div>
    )
}

export default InsightNumber