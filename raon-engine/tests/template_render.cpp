// Renders the chat template for a request JSON ({messages, tools?, chat_template_kwargs?}) read from stdin.
#include <iostream>
#include <iterator>

#include "chat.h"

int main() {
    std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    auto req = raon::chat::json::parse(in);
    raon::chat::TemplateOptions opt;
    if (req.contains("chat_template_kwargs")) {
        const auto& k = req["chat_template_kwargs"];
        opt.enable_thinking = k.value("enable_thinking", true);
        opt.truncate_history_thinking = k.value("truncate_history_thinking", false);
    }
    try {
        std::cout << raon::chat::render(req["messages"], req.value("tools", raon::chat::json()), opt);
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what();
        return 2;
    }
    return 0;
}
