import { cleanup, fireEvent, screen } from "@testing-library/react"
import SettingSlider from "components/SettingSlider"
import renderWithProviders from "tests/utils"

afterEach(cleanup);
const settingSlider = jest.fn();
test("render the settingSlider component", () => {

    renderWithProviders(<SettingSlider setSettingSliderShow={settingSlider} />);
    //expect(screen.getByText(/Settings/)).toBeInTheDocument();

    const closeBtn = screen.getByRole("button", { name: 'Cancel' });
    fireEvent.click(closeBtn);
    expect(settingSlider).toBeCalled();
})