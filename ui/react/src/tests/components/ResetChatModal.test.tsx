import { cleanup, fireEvent, screen } from "@testing-library/react";
import ResetChat from "components/ResetChatModal";
import renderWithProviders from "tests/utils";

afterEach(cleanup);

const resetModalShow = true;
const setResetModalShow = jest.fn();

test("render the resetChat", () => {
    renderWithProviders(<ResetChat resetModalShow={resetModalShow} setResetModalShow={setResetModalShow} />)

    const closeBtn = resetModalShow && screen.getByRole("button", { name: 'Cancel' });
    fireEvent.click(closeBtn);

    const resetBtn = resetModalShow && screen.getByRole("button", { name: 'Reset' });
    fireEvent.click(resetBtn);
})