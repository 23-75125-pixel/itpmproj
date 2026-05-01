(function () {
    const BRAND_COLOR = "#8a1538";
    const CANCEL_COLOR = "#64748b";

    function normalizeText(value) {
        return String(value || "").trim();
    }

    function escapeHtml(value) {
        return normalizeText(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function toHtmlBlock(value) {
        const text = normalizeText(value);
        if (!text) {
            return "";
        }
        return `<div style="white-space:pre-line;text-align:left;line-height:1.6;">${escapeHtml(text).replace(/\n/g, "<br>")}</div>`;
    }

    function baseOptions() {
        return {
            confirmButtonColor: BRAND_COLOR,
            cancelButtonColor: CANCEL_COLOR,
            reverseButtons: true,
            focusCancel: true,
            heightAuto: false,
        };
    }

    function buildFallbackMessage(options) {
        return [options.title, options.text]
            .map(normalizeText)
            .filter(Boolean)
            .join("\n\n");
    }

    function datasetOptions(element) {
        if (!element || !element.dataset) {
            return {};
        }

        const options = {};
        if (element.dataset.swalTitle) {
            options.title = element.dataset.swalTitle;
        }
        if (element.dataset.swalText) {
            options.text = element.dataset.swalText;
        }
        if (element.dataset.swalHtml) {
            options.html = element.dataset.swalHtml;
        }
        if (element.dataset.swalIcon) {
            options.icon = element.dataset.swalIcon;
        }
        if (element.dataset.swalConfirmText) {
            options.confirmButtonText = element.dataset.swalConfirmText;
        }
        if (element.dataset.swalCancelText) {
            options.cancelButtonText = element.dataset.swalCancelText;
        }
        if (element.dataset.swalPosition) {
            options.position = element.dataset.swalPosition;
        }
        if (element.dataset.swalTimer) {
            const timer = Number(element.dataset.swalTimer);
            if (!Number.isNaN(timer) && timer > 0) {
                options.timer = timer;
            }
        }
        return options;
    }

    async function fire(options = {}) {
        const merged = { ...baseOptions(), ...options };

        if (merged.text && !merged.html) {
            merged.html = toHtmlBlock(merged.text);
            delete merged.text;
        }

        if (!window.Swal) {
            if (merged.showCancelButton) {
                return { isConfirmed: window.confirm(buildFallbackMessage(options)) };
            }
            window.alert(buildFallbackMessage(options));
            return { isConfirmed: true };
        }

        return window.Swal.fire(merged);
    }

    async function confirm(options = {}) {
        const result = await fire({
            title: "Please confirm",
            text: "Are you sure you want to continue?",
            icon: "warning",
            showCancelButton: true,
            confirmButtonText: "Yes, continue",
            cancelButtonText: "Cancel",
            ...options,
        });
        return Boolean(result && result.isConfirmed);
    }

    async function toast(options = {}) {
        return fire({
            toast: true,
            position: "top-end",
            showConfirmButton: false,
            timer: 2200,
            timerProgressBar: true,
            icon: "success",
            ...options,
        });
    }

    async function success(title, text) {
        return fire({ title, text, icon: "success" });
    }

    async function error(title, text) {
        return fire({ title, text, icon: "error" });
    }

    function hideMessageElement(element) {
        if (!element) {
            return;
        }
        const wrapper = element.closest(".manager-banner");
        if (wrapper) {
            wrapper.hidden = true;
            return;
        }
        element.hidden = true;
    }

    async function showInlineMessages() {
        const elements = Array.from(document.querySelectorAll("[data-swal-message]"));
        for (const element of elements) {
            const inlineOptions = datasetOptions(element);
            const text = normalizeText(element.dataset.swalMessageText || element.textContent);
            if (!text) {
                continue;
            }
            hideMessageElement(element);
            await fire({
                title: inlineOptions.title || (inlineOptions.icon === "error" ? "Something went wrong" : "Success"),
                text,
                confirmButtonText: inlineOptions.confirmButtonText || "OK",
                ...inlineOptions,
            });
        }
    }

    async function showQueryStateMessages() {
        const params = new URLSearchParams(window.location.search);
        let updated = false;

        if (params.get("login_success") === "1") {
            await fire({
                title: "Welcome!",
                text: "You have logged in successfully.",
                icon: "success",
                confirmButtonText: "Continue",
            });
            params.delete("login_success");
            updated = true;
        }

        if (params.get("logout_success") === "1") {
            await fire({
                title: "Logged out",
                text: "You have logged out successfully.",
                icon: "success",
                confirmButtonText: "OK",
            });
            params.delete("logout_success");
            updated = true;
        }

        if (updated) {
            const query = params.toString();
            const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash || ""}`;
            window.history.replaceState({}, document.title, nextUrl);
        }
    }

    function installConfirmationInterceptor() {
        document.addEventListener(
            "submit",
            async (event) => {
                const form = event.target;
                if (!(form instanceof HTMLFormElement)) {
                    return;
                }

                if (form.dataset.swalBypass === "1") {
                    delete form.dataset.swalBypass;
                    return;
                }

                const submitter = event.submitter || null;
                const shouldConfirm = form.hasAttribute("data-swal-confirm") || Boolean(submitter && submitter.hasAttribute("data-swal-confirm"));
                if (!shouldConfirm) {
                    return;
                }

                event.preventDefault();
                const options = {
                    ...datasetOptions(form),
                    ...datasetOptions(submitter),
                };
                const approved = await confirm(options);
                if (!approved) {
                    return;
                }

                form.dataset.swalBypass = "1";
                if (submitter && typeof form.requestSubmit === "function") {
                    form.requestSubmit(submitter);
                    return;
                }
                form.submit();
            },
            true,
        );
    }

    window.semcdsSwal = {
        fire,
        confirm,
        toast,
        success,
        error,
        toHtmlBlock,
        escapeHtml,
    };

    document.addEventListener("DOMContentLoaded", async () => {
        installConfirmationInterceptor();
        await showQueryStateMessages();
        await showInlineMessages();
    });
})();
