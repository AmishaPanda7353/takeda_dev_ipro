import { cleanup, screen } from "@testing-library/react"
import Footer from "components/footer"
import renderWithProviders from "tests/utils"

afterEach(cleanup)

test("render the footer component", () => {
    renderWithProviders(<Footer />);
    // const copyright = screen.getByText(/Copyright 2023. All Rights Reserved/i);
    // expect(copyright).toBeInTheDocument();

    // expect(screen.getByText(/Privacy Policy/i)).toBeInTheDocument();
    // const copyright = screen.getByText(/Copyright 2023. All Rights Reserved /);
    // expect(copyright).toBeInTheDocument();

    // const policy = screen.getByText(/Privacy Policy/i);
    // expect(policy).toBeInTheDocument();

    // const terms = screen.getByText(/Terms of Service/i);
    // expect(terms).toBeInTheDocument();


})