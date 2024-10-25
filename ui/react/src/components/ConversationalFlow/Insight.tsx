/* eslint-disable import/no-extraneous-dependencies */
import React, { useEffect, useState } from 'react';

function Insight({ insightText, errorSummary, botResponseSummary, showErrorSummary }
    :
    { insightText: string, errorSummary: string, botResponseSummary: string, showErrorSummary: boolean }) {
    const [lines, setLines] = useState<string[]>([]);
    const [currentLine, setCurrentLine] = useState(0);

    useEffect(() => {
        const parts = insightText.split(/[\r\n]+/);
        const filteredParts = parts.filter(part => part.trim() !== '');
        setLines(filteredParts);
    }, [insightText]);

    useEffect(() => {
        if (currentLine < lines.length - 1) {
            const timeout = setTimeout(() => {
                setCurrentLine((prevLine) => prevLine + 1);
            }, 5000); // Adjust the delay between lines here (in milliseconds)

            return () => clearTimeout(timeout);
        }
        return undefined;
    }, [currentLine, lines]);

    function replaceButtlet(line: string) {
        return line.replace(/^- /, ' ');  // Replace "-" with an empty bullet (disc)
    }


    return (
        <div>
            {botResponseSummary && botResponseSummary.length > 0 &&
                <div>
                    {botResponseSummary}
                </div>
            }
            {showErrorSummary && errorSummary && errorSummary?.length > 0 && (

                <div className='text-sm' >
                    <b>Error:</b>{errorSummary}
                </div>

            )}
            <div className="font-poppins text-[14px] leading-relaxed">
                <ul style={{ listStyleType: "disc", marginLeft: "1em" }}>
                    {lines.map((line, index) => (
                        <li>
                            {replaceButtlet(line.trim())}
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}

export default Insight;
