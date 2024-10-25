import React from 'react'

function Footer() {
    return (
        <div className='flex bg-[#FFFFFF] justify-center text-xs opacity-[1]'>
            <div className='text-[#C3C3C3] mr-[20px]'>
                Copyright 2023. All Rights Reserved
            </div>
            <span className='mr-[20px]'> | </span>
            <div className='text-[#C3C3C3] mr-[20px]'>
                Privacy Policy
            </div>
            <span className='mr-[20px]'> | </span>
            <div className='text-[#C3C3C3]'>
                Terms of Service
            </div>
        </div>
    )
}

export default Footer