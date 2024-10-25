import React, { useState, useEffect } from 'react';
import { Formik, FormikHelpers, FormikProps, Form, Field, FieldProps } from 'formik';
import _ from 'lodash'
import SlideOver from './SlideOver';

function SettingSlider({ setSettingSliderShow }: { setSettingSliderShow: (flag: boolean) => void }) {

  const dynamicValues = [{
    title: "User Interface",
    options: [{ type: "checkbox", label: "Insights (Data, Graph, Summary)" },
    { type: "checkbox", label: "SQL Code" }]
  },
  {
    title: "Report",
    options: [{ type: "checkbox", label: "Insights (Data, Graph, Summary)" },
    { type: "checkbox", label: "SQL Code" }]
  },

  ]

  const initialValues = {
    "User Interface": [],
    "Report": [],
    "APIKey": '',
  }

  return (
    <SlideOver closeOnClick={() => setSettingSliderShow(false)}>

      <div className="font-poppins font-semibold text-[18px] text-black mb-[25px]">Settings</div>
      <Formik
        initialValues={initialValues}
        onSubmit={(values, actions) => {
          console.log({ values, actions });
          alert(JSON.stringify(values, null, 2));
          actions.setSubmitting(false);
        }}
      >

        {formik =>
        (
          <Form>
            {_.map(dynamicValues, (item, index) =>
            (
              <div className="mb-4" key={index}>
                <div className="mb-4 font-semibold text-[14px] text-black font-poppins">{item.title}</div>
                <div className="flex flex-col">
                  {
                    item.options?.map((option) =>
                    (

                      <div key={option.label} className="flex gap-1 items-center mb-5 text-[14px] font-poppins">
                        <Field type={option.type} id={item.title} name={item.title} value={option.label} className="focus:ring-0" />
                        <div>{option.label}</div>
                      </div>

                    )
                    )
                  }
                </div>
                <hr className="mt-5" />
              </div>
            ))
            }

            <div className="flex flex-col text-[14px] font-poppins">
              <label className=" text-black font-semibold mb-4" htmlFor="APIKey">Add Your API Key:</label>
              <Field className="p-2 border-1 w-full rounded border-[#DEE3EA] opacity-[0.5] " disabled type="text" id="APIKey" name="APIKey" placeholder="AbcdefghX" />
            </div>

            <div className="flex justify-end fixed gap-5 font-medium bottom-3 right-1">
              <button type="button" className="font-poppins cursor-default rounded bg-white text-[#18749C] hover:border-[#18749C] hover:border-[1px] h-[40px] w-[90px] cursor-pointer" onClick={() => setSettingSliderShow(false)}>Cancel</button>
              <button type="submit" disabled={!formik.dirty} className="cursor-pointer font-poppins rounded disabled:opacity-50 disabled:cursor-not-allowed text-white bg-[#18749C] h-[40px] w-[90px]" >Save</button>
            </div>

          </Form>
        )
        }
      </Formik>

    </SlideOver>
  );
}

export default SettingSlider;