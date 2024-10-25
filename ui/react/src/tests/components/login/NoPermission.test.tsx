import NoPermission from "pages/login/NoPermission";
import renderWithProviders from "tests/utils";
import { fireEvent } from "@testing-library/react";
import '@testing-library/jest-dom/extend-expect';

describe("NoPermission", () => {
    test("Render Permission component", () => {
        const { getByAltText, getByTestId, getByText } = renderWithProviders(<NoPermission />);

        // Check whether the elements are available in the component or not
        expect(getByAltText("Insights Pro")).toBeInTheDocument();
        expect(getByTestId("routechange-button")).toBeInTheDocument();
        expect(getByText("You do not have permission to access the application")).toBeInTheDocument();
        expect(getByText("Powered by Tiger Analytics")).toBeInTheDocument();
    });

    // test("redirects to login page when the login button is clicked", () => {
    //     // Helper function to encapsulate redirection logic
    //     const redirectTo = jest.fn();

    //     // Mock the module that contains the redirection logic
    //     jest.mock("/#/insights", () => ({
    //         redirectTo: redirectTo,
    //     }));

    //     const { getByTestId } = renderWithProviders(<NoPermission />);
    //     const loginButton = getByTestId("routechange-button");

    //     fireEvent.click(loginButton);

    //     // Verify that the redirectTo function is called with the correct value
    //     expect(redirectTo).toHaveBeenCalledWith(window.location.origin);
    // });
});
