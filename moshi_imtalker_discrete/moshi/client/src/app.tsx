import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  createHashRouter,
  RouterProvider,
} from "react-router-dom";
import "./index.css";
// @ts-expect-error - Worker is not recognized by the TS compiler
import { DecoderWorker } from "./decoder/decoderWorker";
import { Queue } from "./pages/Queue/Queue";

const routes = [{ path: "/", element: <Queue /> }];

// BrowserRouter only matches the real URL path. At http://host/moshi#/?embed=1 the
// pathname stays /moshi, so "/" never matches and the UI is blank. Hash routing
// reads path + query from the fragment, which is what the embedded launcher uses.
const pathNoTrailing = window.location.pathname.replace(/\/$/, "");
const useHashRouter = pathNoTrailing !== "";
const router = useHashRouter
  ? createHashRouter(routes)
  : createBrowserRouter(routes);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <RouterProvider router={router} />,
);
