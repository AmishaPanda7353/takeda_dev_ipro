import React, { useState } from 'react'

function TableListSelection() {
    const tableData = [{ name: "Table Heading 1" }, { name: "Table Heading 2" }, { name: "Table Heading 3" }, { name: "Table Heading 4" }];
    const [names, setNames] = useState(tableData);



    return (
        <div>
            <span className='text-[#49494A] font-semibold text-sm'>Table List</span>
            <div className='pt-[20px]'>
                {names.map((val) => (
                    <div className='py-[6px]'>
                        <label>
                            <input className='mr-[10px]' type="checkbox" />
                            <span className='text-[#4F4F4F] capitalize'>{val.name}</span>
                        </label>
                    </div >
                ))
                }
            </div >

        </div >
    )
}

export default TableListSelection