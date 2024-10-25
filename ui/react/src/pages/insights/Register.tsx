import RegisterCard from "components/RegisterCard"
import { setUser } from "store/slice/userSlice";
import { useAppDispatch } from "hooks/store-hooks";


export default function Register() {
    const dispatch = useAppDispatch();

    function handleLogin() { dispatch(setUser(true)); }

    return (
        <div className="h-[92%] w-[99%] flex justify-center items-center align-center" onClick={handleLogin}>
            <RegisterCard />
        </div>
    )
}