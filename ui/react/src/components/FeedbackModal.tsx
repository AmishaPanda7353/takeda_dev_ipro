import { Formik, Form, Field } from 'formik';
import _ from 'lodash'
import { Modal } from '@mui/material'
import dislikeIcon from '../assets/images/dislikeIconSmall.png'

type FeedbackProp = {
    feedbackModalShow: boolean;
    setfeedbackModalShow: (flag: boolean) => void;
    setFeedbackText: (flag: boolean, cnt: string) => void;
}

function Feedback({ feedbackModalShow, setfeedbackModalShow, setFeedbackText }: FeedbackProp) {

    const dynamicValues = [{
        title: "Please select your issue from below",
        options: [{ type: "checkbox", label: "Graph is irrelevant" },
        { type: "checkbox", label: "Table is not giving proper information" },
        { type: "checkbox", label: "Summary is inappropriate" },
        ]
    },
    ]

    const initialValues = {
        issue: [],
        description: "",
    }
    return (
        <Modal open={feedbackModalShow}>
            <div className="bg-white absolute rounded-[10px] top-1/2 left-1/2 transform -translate-x-2/4 -translate-y-2/4 p-[20px] w-[436px] h-[453px] outline-none">
                <div className="flex">
                    <img src={dislikeIcon} alt="dislike" />
                    <div className="ml-2 font-poppins font-semibold text-[16px] text-black">Provide Additional Feedback</div>
                </div>

                <Formik
                    initialValues={initialValues}
                    onSubmit={(values) => {
                        // console.log(values);
                        setfeedbackModalShow(false);
                        const feedback = `${values.issue.concat()}, ${values.description}`;
                        // console.log(feedback)
                        setFeedbackText(false, feedback);
                    }}
                >

                    {formik =>
                    (
                        <Form>
                            <div className="mt-[26px]">
                                {_.map(dynamicValues, (item, index) => (
                                    <div className="mb-3 mt-5 font-poppins" key={index}>
                                        <div className="mb-[15px] font-poppins text-[14px] text-black">{item.title}</div>
                                        <div className="flex flex-col">
                                            {
                                                item.options?.map((option) =>
                                                (
                                                    <div key={option.label} className="flex gap-2 text-[14px] text-[#4F4F4F] items-center mb-[15px] font-poppins">
                                                        <Field type={option.type} id='issue' name='issue' value={option.label} className="focus:ring-0" />
                                                        <div>{option?.label}</div>
                                                    </div>
                                                )
                                                )
                                            }
                                        </div>
                                    </div>
                                ))
                                }

                            </div>

                            <div>
                                <label className="font-poppins text-black text-[14px]" htmlFor="description">Tell us more</label>
                                <textarea className="font-poppins text-[14px] mt-4 rounded resize-none w-full border-1 border-slate-300" id="description" name="description" onChange={formik.handleChange} rows={2} placeholder='Describe your feedback...' />
                            </div>

                            <div className="flex justify-end gap-5 mt-[19px] font-medium">
                                <button type="button" className="font-poppins rounded bg-white cursor-pointer text-[#18749C] border-[#18749C] border-[1px] h-[40px] w-[90px]" onClick={() => { setfeedbackModalShow(false) }} >Cancel</button>
                                <button type="submit" className="font-poppins rounded cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed text-white bg-[#18749C] h-[40px] w-[90px]" disabled={!formik.dirty}>Submit</button>
                            </div>
                        </Form>
                    )
                    }

                </Formik>
            </div>
        </Modal>
    )
}

export default Feedback;