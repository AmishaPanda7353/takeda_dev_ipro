import React from 'react';

function BouncingDotsLoader() {
    return (
        <div className="flex items-center justify-center">
            <div className="flex">
                <div className="w-[10px] h-[10px] mr-[5px] bg-gray-700 rounded-full animate-bounce">
                    <div className="w-full h-full bg-gray-200 rounded-full opacity-0 animate-fade-in animate-pulse" />
                </div>
                <div className="w-[10px] h-[10px] mr-[5px] bg-gray-700 rounded-full animate-bounce ">
                    <div className="w-full h-full bg-gray-200 rounded-full opacity-0 animate-fade-in animate-pulse" />
                </div>
                <div className="w-[10px] h-[10px] mr-[5px] bg-gray-700 rounded-full animate-bounce ">
                    <div className="w-full h-full bg-gray-200 rounded-full opacity-0 animate-fade-in animate-pulse" />
                </div>
            </div>
        </div>
    );
}

export default BouncingDotsLoader;
