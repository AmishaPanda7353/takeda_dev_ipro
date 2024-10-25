import LoginCard from "components/LoginCard"
import { setUser } from "store/slice/userSlice";
import { useAppDispatch } from "hooks/store-hooks";


export default function Login() {
    const dispatch = useAppDispatch();

    function handleLogin() { dispatch(setUser(true)); }

    return (
        <div data-testid="login-card" className="w-[100%] h-[100vh] bg-gray-100 flex flex-col">
            <div className="h-[92%] w-[99%] flex justify-center items-center align-center" onClick={handleLogin}>
                <LoginCard />
            </div>
        </div>
    )
}