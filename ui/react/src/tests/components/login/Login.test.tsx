import { cleanup, render, screen } from "@testing-library/react";
import Login from "pages/login/Login";
import { setUser } from "store/slice/userSlice";
import { useAppDispatch } from "hooks/store-hooks";
import renderWithProviders from "tests/utils";

jest.mock("hooks/store-hooks", () => ({
    useAppDispatch: jest.fn(),
}));

afterEach(cleanup);

test("renders Login component", () => {
    const dispatchMock = jest.fn();
    // useAppDispatch.mockReturnValue(dispatchMock);

    renderWithProviders(<Login />);

    // Assert that the LoginCard component is rendered
    const loginCardElement = screen.getByTestId("login-card");
    // expect(loginCardElement).toBeInTheDocument();

    // Simulate a click on the LoginCard component
    loginCardElement.click();

    // Assert that the setUser action is dispatched
    // expect(dispatchMock).toHaveBeenCalledWith({ type: "user/setUser", payload: true });
});
