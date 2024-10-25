/* eslint-disable import/no-extraneous-dependencies */
// import/no-extraneous-dependencies
import React, { useState } from 'react';
import clipboard from 'clipboard-copy';
import { format } from 'sql-formatter';
import ContentCopyOutlinedIcon from '@mui/icons-material/ContentCopyOutlined';

function Query({ query, error, showError, botResponseQuery }: { query: string, error: string, showError: boolean, botResponseQuery: string, }) {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    clipboard(format(query, { language: 'mysql' }));
    setIsCopied(true);
    // Reset the copied state after a certain duration
    setTimeout(() => {
      setIsCopied(false);
    }, 5000);
  };

  const getFormattedQuery = () => {
    if (query === '' || query === null || query === undefined) {
      return <div className='text-sm'>{query}</div>;
    }

    let formattedQuery = '';
    try {
      formattedQuery = format(query, { language: 'mysql', tabWidth: 6 });
    }
    catch (exc) {
      return <div className='text-sm'>{query}</div>;
    }
    const lines = formattedQuery.split('\n');

    return lines.map((line: any, index: any) => {
      // Check if the line contains a specific keyword and apply a color
      let lineStyle = {};
      if (line.includes('SELECT')) {
        lineStyle = { color: 'blue' };
      } else if (line.includes('FROM')) {
        lineStyle = { color: 'green' };
      } else if (line.includes('WHERE')) {
        lineStyle = { color: 'orange' };
      } else if (line.includes('GROUP BY')) {
        lineStyle = { color: '#c92c0f' };
      } else if (line.includes('ORDER BY')) {
        lineStyle = { color: '#129b57' };
      }

      return (
        <div >
          <span style={{ marginRight: '5px', color: '#999', userSelect: 'none' }}>{index + 1}</span>
          <span style={lineStyle}>
            {line}
          </span>
        </div>
      );
    });
  };

  return (
    <div>
      {botResponseQuery && botResponseQuery.length > 0 &&
        <div className='text-sm'>
          {botResponseQuery}
        </div>}
      {showError && error && error?.length > 0 && (
        <div className='text-sm' >
          <b>Error:</b>{error}
        </div>
      )}

      <div className="flex justify-between">
        <div className="text-[14px] font-semibold mb-[15px] font-poppins"> </div>
        <div data-testid='copy-icon'>
          <ContentCopyOutlinedIcon
            onClick={handleCopy}
            style={{ color: isCopied ? 'green' : 'inherit', cursor: 'pointer' }}
          />
          {isCopied && (
            <span style={{ marginLeft: '5px', color: 'green' }} aria-label="copied-text">
              Copied!
            </span>
          )}
        </div>
      </div>
      <pre
        className="font-poppins text-[14px] leading-relaxed"
        style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word', maxWidth: '100%' }}
      >
        {getFormattedQuery()}
      </pre>
    </div>
  );
}

export default Query;
