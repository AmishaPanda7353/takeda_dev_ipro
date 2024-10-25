import { cleanup, fireEvent, screen } from "@testing-library/react"
import TableButtons from "components/table/TableButtons"
import renderWithProviders from "tests/utils"

afterEach(cleanup)

const setModalShow = jest.fn();

test("render tableButton", () => {
    renderWithProviders(<TableButtons setModalShow={setModalShow} text="Save" />)

    const closeBtn = screen.getByRole("button", { name: 'Cancel' });
    fireEvent.click(closeBtn);
    expect(setModalShow).toBeCalled();
})