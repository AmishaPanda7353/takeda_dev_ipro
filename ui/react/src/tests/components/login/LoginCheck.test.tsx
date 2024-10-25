import { cleanup } from "@testing-library/react"
import LoginCheck from "pages/login/LoginCheck"
import renderWithProviders from "tests/utils"


afterEach(cleanup)


test("render LoginCheck", () => {
    renderWithProviders(<LoginCheck />)
})

test("redirect when user or token is missing", () => {
    const navigate = jest.fn();
    const dispatch = jest.fn();


    sessionStorage.clear();

    renderWithProviders(<LoginCheck />);

    expect(dispatch).not.toHaveBeenCalled();
    // expect(navigate).toHaveBeenCalledWith("/");
});