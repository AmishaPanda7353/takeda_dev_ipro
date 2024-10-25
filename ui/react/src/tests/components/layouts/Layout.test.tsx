import { cleanup } from "@testing-library/react"
import Layout from "layouts/Layout"
import renderWithProviders from "tests/utils"

afterEach(cleanup);

test("render Layout", () => {
    renderWithProviders(<Layout />)
})