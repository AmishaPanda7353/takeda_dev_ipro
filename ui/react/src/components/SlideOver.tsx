import React from "react";
import ChevronRightIcon from '@mui/icons-material/ChevronRight';

export type SlideOverProps = {
  children: React.ReactNode;
  closeOnClick: () => void;
  closeIcon?: boolean;
}

function SlideOver({ children, closeOnClick, closeIcon }: SlideOverProps) {
  return (
    <div className="relative z-20" aria-labelledby="slide-over-title" role="dialog" aria-modal="true">
      <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
      <div className="fixed inset-0 overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-md pl-10">
            <div className="pointer-events-auto relative max-w-md w-screen">
              <div className="absolute z-10 mr-5 flex cursor-pointer hover:bg-gray-100 rounded-full right-1 top-2">
              {closeIcon ? <button type="button" className=" rounded-md focus:outline-none p-1 focus:ring-0 focus:ring-white " onClick={() => closeOnClick()} >

                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" aria-hidden="true">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>:<div> </div>}
              </div>
              <div className="flex h-full flex-col  bg-white py-1 pt-0 shadow-xl">
                <div className="overflow-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-white relative mt-4 flex-1 px-4 sm:px-6">
                  {children}
                </div>
              </div>
            </div>
          </div>
        </div >
      </div >
    </div >
  )
}

SlideOver.defaultProps = {
  closeIcon: false,
}

export default SlideOver;