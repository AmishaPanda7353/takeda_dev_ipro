import { cleanup, fireEvent, screen } from "@testing-library/react"
import ChatTable from "components/table"
import renderWithProviders from "tests/utils"
import { Modal } from "@mui/material";
import userEvent from '@testing-library/user-event'

const data = [[
    { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
]]


describe("ChatTable", () => {
    afterEach(cleanup)



    test("render table", () => {
        let modalShow = true
        const setModalShow = jest.fn((value) => {
            modalShow = value;
        });
        // renderWithProviders(<ChatTable setModalShow={setModalShow} modalShow={modalShow} data={data} />);

        // modal content
        // expect(screen.getByText("Please select the most relevant tables for answering your question")).toBeInTheDocument();
        // expect(screen.getByText("Continue")).toBeInTheDocument();

    })



})
