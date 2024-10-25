import React from 'react'

export type AnswerLayoutProps = {
    children: React.ReactNode;
}

function AnswerLayout({ children }: AnswerLayoutProps) {
    return (
        <div>
            <div className='bg-[#FFFFFF] rounded-[10px] opacity-[1px] m-[20px]'>
                {children}
            </div>
        </div>
    )
}

export default AnswerLayout