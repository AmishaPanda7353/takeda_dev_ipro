import React from 'react'

export type QuestionLayoutProps = {
    children: React.ReactNode;
}

function QuestionLayout({ children }: QuestionLayoutProps) {
    return (
        <div>{children}</div>
    )
}

export default QuestionLayout